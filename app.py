"""Streamlit dashboard for the FlipItNews RNN classifier."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from news_utils import (
    ArtifactError,
    clean_text,
    load_artifacts,
    predict_proba,
    preprocess,
)


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "model_artifacts"
DEFAULT_THRESHOLD = 0.70


st.set_page_config(
    page_title="FlipItNews | AI News Categoriser",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #102a43 0%, #1f4e79 58%, #2f80a8 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.15rem; }
    .hero p { margin: .45rem 0 0; color: #d9efff; font-size: 1.02rem; }
    .section-label {
        color: #2f80a8;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .impact-card {
        border: 1px solid #d9e2ec;
        border-radius: 14px;
        padding: .9rem 1rem;
        background: #f7fafc;
        min-height: 124px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained BiLSTM model …")
def get_artifacts() -> dict:
    return load_artifacts(ARTIFACT_DIR)


@st.cache_data
def get_json(filename: str):
    path = ARTIFACT_DIR / filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


try:
    artifacts = get_artifacts()
except (ArtifactError, ImportError, OSError) as exc:
    st.error("The deployment bundle is incomplete or could not be loaded.")
    st.code(str(exc))
    st.info("Run `python build_artifacts.py` once, then restart Streamlit.")
    st.stop()


classes = artifacts["classes"]
metrics = get_json("metrics.json") or {}
samples = get_json("samples.json") or {}

with st.sidebar:
    st.markdown("### FlipItNews")
    st.caption("Editorial intelligence for faster filing and more relevant discovery.")
    st.divider()
    st.markdown("#### Model health")
    deployed = metrics.get("deployed", {})
    metric_left, metric_right = st.columns(2)
    metric_left.metric("Test accuracy", f"{deployed.get('accuracy', 0):.1%}")
    metric_right.metric("Weighted F1", f"{deployed.get('f1', 0):.1%}")
    threshold = st.slider(
        "Auto-tag confidence threshold",
        min_value=0.30,
        max_value=0.95,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Below this confidence, route the article to an editor for review.",
    )
    st.markdown("**Categories**")
    st.caption(" · ".join(category.title() for category in classes))
    st.divider()
    st.caption("BiLSTM + Word2Vec embeddings · 353-token sequence window")


st.markdown(
    """
    <div class="hero">
      <h1>📰 FlipItNews AI News Categoriser</h1>
      <p>Classify, organise, and route news stories with a confidence-aware recurrent neural network.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_classify, tab_batch, tab_performance, tab_method = st.tabs(
    ["🔎 Classify an article", "🗂 Batch classify", "📊 Model performance", "🧭 How it works"]
)


with tab_classify:
    st.markdown('<div class="section-label">Single-story workflow</div>', unsafe_allow_html=True)
    st.subheader("Turn an unfiled story into a newsroom-ready tag")
    st.write(
        "Paste a headline and article body. The model returns the most likely category, "
        "a probability profile, and an editorial review recommendation."
    )

    if samples:
        st.markdown("**Try a real held-out sample**")
        sample_columns = st.columns(len(samples))

        def choose_sample(value: str) -> None:
            st.session_state["article_text"] = value

        for column, (category, sample_text) in zip(sample_columns, samples.items()):
            column.button(
                category.title(),
                key=f"sample_{category}",
                on_click=choose_sample,
                args=(sample_text,),
                use_container_width=True,
            )

    article_text = st.text_area(
        "Headline + article body",
        key="article_text",
        height=240,
        placeholder="Example: The central bank held interest rates steady as inflation eased …",
    )

    classify = st.button("Classify article", type="primary", use_container_width=False)
    if classify:
        cleaned_tokens = preprocess(article_text, artifacts["stopwords"])
        if len(cleaned_tokens) < 5:
            st.warning("Please provide at least five meaningful words so the model has enough context.")
        else:
            with st.spinner("Reading the story …"):
                probabilities = predict_proba([article_text], artifacts)[0]
            order = np.argsort(probabilities)[::-1]
            predicted = classes[int(order[0])]
            confidence = float(probabilities[order[0]])
            runner_up = classes[int(order[1])] if len(order) > 1 else None

            st.divider()
            result_left, result_right = st.columns([0.9, 1.4])
            with result_left:
                st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
                st.subheader(predicted.title())
                st.metric("Model confidence", f"{confidence:.1%}")
                st.progress(min(confidence, 1.0), text=f"{confidence:.1%} confidence")
                if confidence >= threshold:
                    st.success("Auto-tag recommended — confidence clears the editorial threshold.")
                else:
                    st.warning(
                        f"Editor review recommended — runner-up: {runner_up.title()} "
                        f"({probabilities[order[1]]:.1%})."
                    )
                st.caption(f"{len(cleaned_tokens):,} meaningful tokens after preprocessing.")

            with result_right:
                st.markdown('<div class="section-label">Probability profile</div>', unsafe_allow_html=True)
                chart = pd.DataFrame(
                    {"Category": classes, "Probability": probabilities}
                ).set_index("Category")
                st.bar_chart(chart, y="Probability", color="#2f80a8")
                top_three = pd.DataFrame(
                    {
                        "Category": [classes[int(i)].title() for i in order[:3]],
                        "Probability": [float(probabilities[i]) for i in order[:3]],
                    }
                )
                st.dataframe(
                    top_three.style.format({"Probability": "{:.1%}"}),
                    hide_index=True,
                    use_container_width=True,
                )

            with st.expander("Inspect preprocessing"):
                st.write("The same preprocessing used during training is applied at inference time.")
                st.code(clean_text(article_text)[:1500] or "(empty)", language="text")
                st.caption("First 40 tokens after stop-word removal")
                st.write(" · ".join(cleaned_tokens[:40]) or "(none)")


with tab_batch:
    st.markdown('<div class="section-label">Editorial operations</div>', unsafe_allow_html=True)
    st.subheader("Classify an entire content queue")
    st.write(
        "Upload a CSV, TSV, or text file. The app adds category, confidence, and review-routing columns "
        "so the result can flow into a content-management workflow."
    )
    upload = st.file_uploader("Upload article data", type=["csv", "tsv", "txt"])
    if upload is not None:
        try:
            dataframe = pd.read_csv(io.BytesIO(upload.getvalue()), sep=None, engine="python")
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            dataframe = None

        if dataframe is not None and not dataframe.empty:
            st.write(f"Loaded **{len(dataframe):,}** rows and **{len(dataframe.columns)}** columns.")
            preferred = next(
                (
                    index
                    for index, column in enumerate(dataframe.columns)
                    if str(column).lower() in {"content", "text", "article", "body", "description"}
                ),
                0,
            )
            text_column = st.selectbox("Column containing article text", dataframe.columns, index=preferred)
            if st.button("Classify all rows", type="primary"):
                texts = dataframe[text_column].fillna("").astype(str).tolist()
                with st.spinner(f"Classifying {len(texts):,} articles …"):
                    batch_probabilities = predict_proba(texts, artifacts)
                batch_output = dataframe.copy()
                best_indices = batch_probabilities.argmax(axis=1)
                batch_output["predicted_category"] = [classes[int(i)] for i in best_indices]
                batch_output["confidence"] = batch_probabilities.max(axis=1)
                batch_output["needs_editor_review"] = batch_output["confidence"] < threshold
                st.session_state["batch_output"] = batch_output

            batch_output = st.session_state.get("batch_output")
            if batch_output is not None:
                preview = batch_output.copy()
                preview["confidence"] = preview["confidence"].map(lambda value: f"{value:.1%}")
                st.dataframe(preview.head(200), use_container_width=True, hide_index=True)
                batch_left, batch_right = st.columns([1.3, 1])
                with batch_left:
                    st.markdown("**Predicted category mix**")
                    st.bar_chart(batch_output["predicted_category"].value_counts(), color="#2f80a8")
                with batch_right:
                    review_count = int(batch_output["needs_editor_review"].sum())
                    st.metric("Editor review queue", f"{review_count:,} / {len(batch_output):,}")
                    st.caption(f"{len(batch_output) - review_count:,} rows clear the {threshold:.0%} threshold.")
                st.download_button(
                    "Download classified CSV",
                    batch_output.to_csv(index=False).encode("utf-8"),
                    file_name="flipitnews_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
        elif dataframe is not None:
            st.warning("The uploaded file contains no rows.")


with tab_performance:
    st.markdown('<div class="section-label">Evidence and evaluation</div>', unsafe_allow_html=True)
    st.subheader("Model performance and dataset health")
    dataset = metrics.get("dataset", {})
    preprocessing = metrics.get("preprocessing", {})
    summary = st.columns(4)
    summary[0].metric("Raw articles", f"{dataset.get('raw_rows', 0):,}")
    summary[1].metric("After deduplication", f"{dataset.get('rows_after_deduplication', 0):,}")
    summary[2].metric("Missing values", f"{dataset.get('missing_values', 0):,}")
    summary[3].metric("Vocabulary", f"{preprocessing.get('vocabulary_size', 0):,}")

    category_counts = dataset.get("category_counts", {})
    if category_counts:
        st.markdown("**News category distribution**")
        st.bar_chart(pd.Series(category_counts, name="articles"), color="#2f80a8")

    results = metrics.get("results", [])
    if results:
        st.markdown("**Held-out test-set comparison**")
        result_table = pd.DataFrame(results).set_index("Model")
        metric_columns = ["Accuracy", "Precision", "Recall", "F1-score", "Macro F1"]
        st.dataframe(
            result_table.style.format({column: "{:.1%}" for column in metric_columns if column in result_table})
            .highlight_max(subset=[column for column in ["Accuracy", "F1-score"] if column in result_table], color="#d7f3e3"),
            use_container_width=True,
        )
        plot_columns = [column for column in metric_columns if column in result_table]
        st.bar_chart(result_table[plot_columns], color=["#2f80a8", "#56b4d3", "#7ac7c9", "#2f80a8", "#1b4965"])

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown("**Deployed model — per-category report**")
        report = deployed.get("report", {})
        report_rows = {
            category.title(): values
            for category, values in report.items()
            if category in classes
        }
        if report_rows:
            report_frame = pd.DataFrame(report_rows).T
            show_columns = [column for column in ["precision", "recall", "f1-score", "support"] if column in report_frame]
            st.dataframe(
                report_frame[show_columns].style.format(
                    {column: "{:.1%}" for column in ["precision", "recall", "f1-score"] if column in show_columns}
                ),
                use_container_width=True,
            )
    with lower_right:
        confusion_matrix_path = ARTIFACT_DIR / "confusion_matrix_rnn.png"
        if confusion_matrix_path.exists():
            st.image(str(confusion_matrix_path), caption="BiLSTM confusion matrix", use_container_width=True)


with tab_method:
    st.markdown('<div class="section-label">From raw text to personalised delivery</div>', unsafe_allow_html=True)
    st.subheader("A practical human-in-the-loop content pipeline")
    st.write(
        "FlipItNews can use the predicted category as a first-class content attribute: it improves filing, "
        "search facets, related-story retrieval, and category-aware recommendation rules."
    )
    cards = st.columns(3)
    cards[0].markdown(
        '<div class="impact-card"><b>01 · Normalise</b><br>Lowercase, remove punctuation and stop words, then tokenize the story.</div>',
        unsafe_allow_html=True,
    )
    cards[1].markdown(
        '<div class="impact-card"><b>02 · Understand</b><br>Word2Vec-initialised embeddings preserve sequence context for the BiLSTM classifier.</div>',
        unsafe_allow_html=True,
    )
    cards[2].markdown(
        '<div class="impact-card"><b>03 · Route</b><br>High-confidence tags can be automated; uncertain stories are surfaced for editors.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("**Why this helps FlipItNews**")
    st.markdown(
        "- **Content organisation:** consistent category tags reduce manual filing and make archives easier to search.\n"
        "- **Personalised delivery:** category signals can power topic follows, newsletter sections, and reader-specific feeds.\n"
        "- **Operational focus:** at a 70% confidence threshold, the benchmark routed about 96.9% of test stories to auto-tagging and reserved the ambiguous tail for editors.\n"
        "- **Responsible deployment:** confidence is exposed to editors, and low-confidence predictions remain reviewable rather than silently published."
    )
    st.caption(
        "The model is trained on the BBC News topic dataset and should be monitored for source, language, "
        "and topic drift before being used as the sole publishing decision."
    )
