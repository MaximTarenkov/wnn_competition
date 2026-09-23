"""Параллельная валидация модели на всех доступных ядрах CPU (12 cores)."""
from pathlib import Path
import os
import sys
import time
import argparse
import multiprocessing as mp
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# Ограничиваем каждую дочернюю сессию внутри строго 1 потоком
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Подключаем utils.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from utils import (
    GlobalAccumulator,
    validate_sequence,
    DataPoint,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    SEQUENCE_LENGTH,
)
from solution import PredictionModel

# Глобальные переменные для каждого отдельного процесса-воркера
_worker_model = None
_worker_parquet = None


def worker_init(parquet_path: str):
    """Инициализация воркера: каждый процесс держит свой инстанс модели и читатель parquet."""
    global _worker_model, _worker_parquet
    _worker_parquet = pq.ParquetFile(parquet_path)
    _worker_model = PredictionModel()


def process_sequence(group_idx: int):
    """Обработка ровно одной 20k-последовательности одним ядром."""
    global _worker_model, _worker_parquet

    table = _worker_parquet.read_row_group(group_idx, use_threads=False)
    seq, need = validate_sequence(table)

    features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    targets = np.column_stack([table[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
    predictions = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)

    # Пошаговый инференс внутри процесса
    for step in range(SEQUENCE_LENGTH):
        point = DataPoint(seq, step, bool(need[step]), features[step])
        val = _worker_model.predict(point)
        if need[step]:
            predictions[step] = val

    # Накапливаем статистики внутри последовательности
    mask = table["is_scored"].to_numpy().astype(bool) & need
    local_acc = GlobalAccumulator()
    local_acc.add(targets, predictions, mask)

    # Возвращаем накопленные достаточные статистики
    return {
        "blocks_seen": local_acc.blocks_seen,
        "selected_rows": local_acc.selected_rows,
        "sum_w": local_acc.sum_w,
        "sum_wy": local_acc.sum_wy,
        "sum_wp": local_acc.sum_wp,
        "sum_wyy": local_acc.sum_wyy,
        "sum_wpp": local_acc.sum_wpp,
        "sum_wyp": local_acc.sum_wyp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default="../datasets/valid.parquet", help="Путь к parquet")
    parser.add_argument("--n_workers", type=int, default=12, help="Число ядер CPU")
    args = parser.parse_args()

    n_workers = min(args.n_workers, os.cpu_count() or 1)
    parquet = pq.ParquetFile(args.validation)
    num_sequences = parquet.num_row_groups
    total_steps = num_sequences * SEQUENCE_LENGTH

    print("=" * 65)
    print(f"СТАРТ ПАРАЛЛЕЛЬНОГО ТЕСТИРОВАНИЯ НА {n_workers} ЯДРАХ CPU")
    print(f"Файл: {args.validation}")
    print(f"Последовательностей: {num_sequences:,} | Всего шагов: {total_steps:,}")
    print("=" * 65)

    start_time = time.perf_counter()

    # Создаем контекст multiprocessing ('spawn' обеспечивает чистую инициализацию ONNX)
    ctx = mp.get_context("spawn")
    global_acc = GlobalAccumulator()

    with ctx.Pool(
        processes=n_workers,
        initializer=worker_init,
        initargs=(args.validation,)
    ) as pool:

        # imap_unordered распределяет задачи по ядрам без блокировки
        iterator = pool.imap_unordered(process_sequence, range(num_sequences), chunksize=2)

        with tqdm(total=num_sequences, desc=f"Прогресс ({n_workers} CPU)", unit=" seq", dynamic_ncols=True) as pbar:
            for stats in iterator:
                # Мгновенно объединяем статистики со всех 12 ядер
                global_acc.blocks_seen += stats["blocks_seen"]
                global_acc.selected_rows += stats["selected_rows"]
                global_acc.sum_w += stats["sum_w"]
                global_acc.sum_wy += stats["sum_wy"]
                global_acc.sum_wp += stats["sum_wp"]
                global_acc.sum_wyy += stats["sum_wyy"]
                global_acc.sum_wpp += stats["sum_wpp"]
                global_acc.sum_wyp += stats["sum_wyp"]

                pbar.update(1)

    total_time = time.perf_counter() - start_time
    result = global_acc.result()
    steps_per_sec = total_steps / total_time

    print("\n" + "=" * 65)
    print("ИТОГОВЫЙ РЕЗУЛЬТАТ (МАТЕМАТИЧЕСКИ ИДЕНТИЧЕН 1 vCPU):")
    print("=" * 65)
    print(f"Global Weighted Pearson:   {result['global_weighted_pearson']:.6f}")
    print(f"  - Таргет t0:             {result['t0']:.6f}")
    print(f"  - Таргет t1:             {result['t1']:.6f}")
    print("-" * 65)
    print(f"Затраченное время:         {total_time:.2f} сек ({total_time / 60:.2f} мин)")
    print(f"Суммарная скорость:        {steps_per_sec:,.0f} шагов/сек")
    print(f"Обработано строк:          {global_acc.selected_rows:,}")
    print("=" * 65)


if __name__ == "__main__":
    main()