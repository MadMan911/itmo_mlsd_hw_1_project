"""EDA from committed diagnostics; no network or full source text required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(snapshot: str) -> dict:
    source = ROOT / "reports/data" / snapshot
    figures = ROOT / "reports/figures"
    figures.mkdir(exist_ok=True)
    docs = pd.read_csv(source / "documents.csv", keep_default_na=False)
    chunks = pd.read_csv(source / "chunks.csv", keep_default_na=False)
    sizes = pd.read_csv(source / "chunking_comparison.csv")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figures / name, dpi=125, bbox_inches="tight", facecolor="white")
        plt.close()

    _fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    counts = docs["format"].value_counts()
    axes[0].bar(counts.index, counts.values, color=["#266e9e", "#ec9b42"])
    axes[0].set(title="Состав корпуса", ylabel="Документы")
    for fmt, group in docs.groupby("format"):
        axes[1].hist(group["words"], bins=15, alpha=0.65, label=fmt)
    axes[1].set(
        xscale="symlog",
        xlabel="Слова (логарифмическая шкала)",
        ylabel="Документы",
        title="Длины документов неоднородны",
    )
    axes[1].legend()
    save("01_composition.png")

    _fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    docs["topic"].value_counts().sort_values().plot.barh(ax=axes[0], color="#266e9e")
    axes[0].set(title="Темы: по правилу отбора источников", xlabel="Документы", ylabel="")
    docs.groupby("topic")["words"].sum().sort_values().plot.barh(ax=axes[1], color="#479b87")
    axes[1].set(title="Те же темы по объёму текста", xlabel="Слова", ylabel="")
    save("02_topics.png")

    _fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    years = docs["year_mentions"].str.split(";").explode()
    years = years[years != ""].value_counts().sort_index()
    axes[0].bar(years.index, years.values, color="#266e9e")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].set(title="Годы в тексте: упоминания, не актуальность", ylabel="Документы")
    docs["freshness"].value_counts().plot.barh(ax=axes[1], color="#ec9b42")
    axes[1].set(title="Ручная проверка области действия", xlabel="Документы", ylabel="")
    save("03_freshness.png")

    quality = pd.Series(
        {
            "Без явного года кампании": (docs["campaign_year"] == "").sum(),
            "Смешанные уровни образования": (docs["education_level"] == "mixed").sum(),
            "Повторные строки ≥ 10% текста": (docs["repeated_line_char_fraction"] >= 0.1).sum(),
            "Мало текста (<80 слов)": docs["low_text"].sum(),
            "Точные дубликаты документов": (docs["exact_duplicate_of"] != "").sum(),
            "Страницы PDF с <50 символами": docs["empty_pdf_pages"].sum(),
            "Есть публичные email": (docs["email_count"] > 0).sum(),
        }
    ).astype(int)
    plt.figure(figsize=(9, 4))
    quality.iloc[::-1].plot.barh(color="#d18246")
    plt.title("Флаги качества: не все означают ошибку")
    plt.xlabel("Количество (PDF-флаг считает страницы)")
    save("04_quality.png")

    _fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    repeated = docs.sort_values("repeated_line_char_fraction", ascending=False).head(8)
    labels = repeated["url"].str.replace("https://abit.itmo.ru", "", regex=False)
    axes[0].barh(
        labels.iloc[::-1], repeated["repeated_line_char_fraction"].iloc[::-1] * 100, color="#266e9e"
    )
    axes[0].set(title="Повторы длинных строк внутри HTML/PDF", xlabel="% символов текста")
    axes[1].hist(chunks["words"], bins=18, color="#479b87")
    axes[1].set(title="Фрагменты: 180 слов, overlap 30", xlabel="Слова", ylabel="Фрагменты")
    save("05_repetition_chunks.png")

    plt.figure(figsize=(6, 3.5))
    labels = [f"{r.size_words} / {r.overlap_words}" for r in sizes.itertuples()]
    plt.bar(labels, sizes["chunks"], color="#266e9e")
    plt.title("Цена размера окна: число фрагментов всего корпуса")
    plt.xlabel("Размер / перекрытие, слова (не токены LLM)")
    plt.ylabel("Фрагменты")
    save("06_chunking_tradeoff.png")

    summary = {
        "snapshot": snapshot,
        "documents": len(docs),
        "formats": counts.to_dict(),
        "bytes": int(docs["bytes"].sum()),
        "words": int(docs["words"].sum()),
        "chars": int(docs["chars"].sum()),
        "chunks": len(chunks),
        "eligible_documents": int(docs["retrieval_eligible"].sum()),
        "eligible_chunks": int(chunks["retrieval_eligible"].sum()),
        "pdf_pages": int(docs["pdf_pages"].sum()),
        "word_quantiles": docs["words"].quantile([0, 0.25, 0.5, 0.75, 0.95, 1]).to_dict(),
        "topics": docs["topic"].value_counts().to_dict(),
        "quality_flags": quality.to_dict(),
        "raw_body_to_text_char_ratio_median": float(
            (docs["full_chars"] / docs["chars"].clip(lower=1)).median()
        ),
        "interpretation": "Purposive snapshot; not a random sample of user requests",
    }
    (source / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="20261111_itmo_bachelor_v1")
    run(parser.parse_args().snapshot)
