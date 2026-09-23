import io
import os
import time
import pyarrow.parquet as pq

file_path = "datasets/train.parquet"
parquet_file = pq.ParquetFile(file_path)
metadata = parquet_file.metadata

TARGET_UNCOMPRESSED_BYTES = 8 * 1024**3  # 8 Гигабайт

# 1. Считаем, сколько Row Groups нужно взять
accumulated_uncompressed = 0
accumulated_compressed = 0
row_groups_to_read = []

for i in range(metadata.num_row_groups):
    rg = metadata.row_group(i)
    rg_uncompressed = rg.total_byte_size
    # Считаем сжатый размер всех колонок в этой группе
    rg_compressed = sum(
        rg.column(col_idx).total_compressed_size
        for col_idx in range(rg.num_columns)
    )

    row_groups_to_read.append(i)
    accumulated_uncompressed += rg_uncompressed
    accumulated_compressed += rg_compressed

    if accumulated_uncompressed >= TARGET_UNCOMPRESSED_BYTES:
        break

print("=" * 50)
print(f"Выбрано групп строк (Row Groups): {len(row_groups_to_read)}")
print(
    f"1. ПРОЦЕССОР распакует (Uncompressed): {accumulated_uncompressed / (1024**3):.2f} GB"
)
print(
    f"2. ДИСК физически прочитает (Compressed): {accumulated_compressed / (1024**3):.2f} GB"
)
print(
    f"Коэффициент сжатия: {accumulated_uncompressed / accumulated_compressed:.2f}x"
)
print("=" * 50)

# ---------------------------------------------------------
# ТЕСТ 1: Чистая скорость чтения диска (I/O)
# Читаем ровно столько байт, сколько весят сжатые данные
# ---------------------------------------------------------
print("\n[1/3] Замеряем чистую скорость диска...")
t0 = time.perf_counter()
with open(file_path, "rb") as f:
    # Читаем блоками по 64 МБ, чтобы не забить сразу всю RAM
    bytes_read = 0
    chunk_size = 64 * 1024 * 1024
    while bytes_read < accumulated_compressed:
        to_read = min(chunk_size, accumulated_compressed - bytes_read)
        chunk = f.read(to_read)
        if not chunk:
            break
        bytes_read += len(chunk)
disk_time = time.perf_counter() - t0
disk_speed = (accumulated_compressed / (1024**2)) / disk_time
print(
    f"Диск прочитал {accumulated_compressed / 1024**3:.2f} GB за {disk_time:.2f} сек ({disk_speed:.1f} MB/s)"
)

# ---------------------------------------------------------
# ТЕСТ 2: Чтение + Декомпрессия процессором в Arrow Table
# ---------------------------------------------------------
print("\n[2/3] Чтение с декомпрессией (Диск + CPU распаковка)...")
t0 = time.perf_counter()
# Читаем только нужные row groups
arrow_table = parquet_file.read_row_groups(row_groups_to_read)
total_read_time = time.perf_counter() - t0

# Примерная оценка времени чисто процессора на распаковку:
cpu_decompress_time = max(0.0, total_read_time - disk_time)
print(f"Всего чтение и декомпрессия: {total_read_time:.2f} сек")
print(
    f" -> Чистая работа CPU по декомпрессии (ориентировочно): ~{cpu_decompress_time:.2f} сек"
)

# ---------------------------------------------------------
# ТЕСТ 3: Конвертация в Pandas (Чистая работа CPU)
# ОСТОРОЖНО: Требует много оперативной памяти!
# ---------------------------------------------------------
print("\n[3/3] Конвертация Arrow -> Pandas DataFrame (CPU)...")
t0 = time.perf_counter()
df = arrow_table.to_pandas(self_destruct=True)
del arrow_table
pandas_time = time.perf_counter() - t0
print(f"Преобразование в Pandas заняло: {pandas_time:.2f} сек")

print("=" * 50)
print(f"ИТОГОВОЕ ВРЕМЯ (Диск + Распаковка + Pandas):")
print(f"{total_read_time + pandas_time:.2f} сек")
print("=" * 50)
