import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import faiss
import numpy as np
import datetime
import streamlit.components.v1 as components
import networkx as nx  # NEW: for network graphs

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Reddit Domain Intel AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main {
        background-color: #f9f9f9;
    }
    .stMetric {
        background-color: #26262f;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED RESOURCE LOADING
# ==========================================

@st.cache_resource
def load_nlp_models():
    """Loads heavy NLP models once and caches them."""
    with st.spinner("Loading NLP Models (Embeddings, VADER, Toxicity, Emotion)..."):
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        sentiment_model = SentimentIntensityAnalyzer()
        try:
            toxicity_pipe = pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=False)
            emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
        except Exception as e:
            st.error(f"Error loading Transformers: {e}. Running in lightweight mode.")
            toxicity_pipe = None
            emotion_pipe = None
    return embedder, sentiment_model, toxicity_pipe, emotion_pipe

@st.cache_data
def load_data(uploaded_file):
    """Loads and preprocesses data."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'created_utc' in df.columns:
                df['created_utc'] = pd.to_datetime(df['created_utc'])
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    return None

def generate_dummy_data():
    """Generates sample data for demonstration if no file is uploaded."""
    dates = pd.date_range(start="2025-01-01", periods=100)
    data = {
        'title': [f"Discussion about topic {i} regarding politics and tech" for i in range(100)],
        'domain': np.random.choice(['youtube.com', 'cnn.com', 'foxnews.com', 'medium.com'], 100),
        'subreddit': np.random.choice(['politics', 'technology', 'worldnews', 'science'], 100),
        'created_utc': dates,
        'url': [f"http://example.com/{i}" for i in range(100)]
    }
    return pd.DataFrame(data)

# ==========================================
# 3. LOGIC & PROCESSING
# ==========================================

def compute_features(df, sentiment_analyzer, toxic_pipe, emo_pipe):
    """Computes Sentiment, Toxicity, and Emotion if columns don't exist."""
    df_proc = df.copy()
    
    # Sentiment
    if 'sentiment' not in df_proc.columns:
        with st.spinner("Computing VADER Sentiment..."):
            df_proc['sentiment'] = df_proc['title'].apply(lambda x: sentiment_analyzer.polarity_scores(str(x))['compound'])
    
    # Toxicity & Emotion (Heavy Compute)
    texts = df_proc['title'].astype(str).tolist()
    
    if 'toxicity' not in df_proc.columns and toxic_pipe:
        with st.spinner("Computing Toxicity Scores (BERT)..."):
            results = toxic_pipe(texts, batch_size=8, truncation=True, max_length=512)
            df_proc['toxicity'] = [r['score'] for r in results]
    elif 'toxicity' not in df_proc.columns:
        df_proc['toxicity'] = 0.0  # Fallback

    if 'emotion' not in df_proc.columns and emo_pipe:
        with st.spinner("Classifying Emotions (RoBERTa)..."):
            results = emo_pipe(texts, batch_size=8, truncation=True, max_length=512)
            df_proc['emotion'] = [r['label'] for r in results]
    elif 'emotion' not in df_proc.columns:
        df_proc['emotion'] = 'neutral'  # Fallback

    return df_proc

def perform_clustering(df_subset):
    """Performs KMeans clustering on titles."""
    if len(df_subset) < 3:
        return df_subset
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(df_subset['title'])
    kmeans = KMeans(n_clusters=min(3, len(df_subset)), random_state=42)
    df_subset = df_subset.copy()
    df_subset['cluster'] = kmeans.fit_predict(X)
    return df_subset

def get_gemini_response(prompt, api_key):
    """Calls Google Gemini API."""
    if not api_key:
        return "No Gemini API key provided. Please add it in the sidebar."
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"

# ========= NETWORK LOGIC (NEW) =========

def build_network_graph(df, mode, selected_domain=None, min_edge_weight=1):
    """
    Builds an interactive network for different relationship types.

    mode:
        - 'subreddit_cluster': Subreddit ↔ Narrative Cluster (per domain)
        - 'subreddit_emotion': Subreddit ↔ Emotion (per domain)
        - 'domain_subreddit': Domain ↔ Subreddit (global)
    """
    # Guard: need basic columns
    required_cols = {'subreddit', 'domain', 'title'}
    if not required_cols.issubset(df.columns):
        return None, pd.DataFrame(), pd.DataFrame()

    df_net = df.copy()

    # Restrict to selected domain where appropriate
    if mode in ['subreddit_cluster', 'subreddit_emotion'] and selected_domain:
        df_net = df_net[df_net['domain'].str.contains(selected_domain, case=False, na=False)]

    if df_net.empty:
        return None, pd.DataFrame(), pd.DataFrame()

    # Build aggregation based on mode
    if mode == 'subreddit_cluster':
        df_net = perform_clustering(df_net)
        if 'cluster' not in df_net.columns:
            return None, pd.DataFrame(), pd.DataFrame()
        agg = (
            df_net.groupby(['subreddit', 'cluster'])
                  .size()
                  .reset_index(name='count')
        )
        agg = agg[agg['count'] >= min_edge_weight]
        if agg.empty:
            return None, pd.DataFrame(), pd.DataFrame()

        G = nx.Graph()
        # Add nodes
        for s in agg['subreddit'].unique():
            total_posts = agg.loc[agg['subreddit'] == s, 'count'].sum()
            G.add_node(s, type='subreddit', size=total_posts)

        for c in agg['cluster'].unique():
            total_posts = agg.loc[agg['cluster'] == c, 'count'].sum()
            node_name = f"Cluster {c}"
            G.add_node(node_name, type='cluster', size=total_posts, cluster_id=int(c))

        # Add edges
        for _, row in agg.iterrows():
            subreddit = row['subreddit']
            cluster_node = f"Cluster {int(row['cluster'])}"
            weight = int(row['count'])
            G.add_edge(subreddit, cluster_node, weight=weight)

    elif mode == 'subreddit_emotion':
        if 'emotion' not in df_net.columns:
            return None, pd.DataFrame(), pd.DataFrame()
        agg = (
            df_net.groupby(['subreddit', 'emotion'])
                  .size()
                  .reset_index(name='count')
        )
        agg = agg[agg['count'] >= min_edge_weight]
        if agg.empty:
            return None, pd.DataFrame(), pd.DataFrame()

        G = nx.Graph()
        for s in agg['subreddit'].unique():
            total_posts = agg.loc[agg['subreddit'] == s, 'count'].sum()
            G.add_node(s, type='subreddit', size=total_posts)

        for emo in agg['emotion'].unique():
            total_posts = agg.loc[agg['emotion'] == emo, 'count'].sum()
            node_name = f"Emotion: {emo}"
            G.add_node(node_name, type='emotion', size=total_posts, emotion=emo)

        for _, row in agg.iterrows():
            subreddit = row['subreddit']
            emotion_node = f"Emotion: {row['emotion']}"
            weight = int(row['count'])
            G.add_edge(subreddit, emotion_node, weight=weight)

    elif mode == 'domain_subreddit':
        agg = (
            df_net.groupby(['domain', 'subreddit'])
                  .size()
                  .reset_index(name='count')
        )
        agg = agg[agg['count'] >= min_edge_weight]
        if agg.empty:
            return None, pd.DataFrame(), pd.DataFrame()

        G = nx.Graph()
        for dname in agg['domain'].unique():
            total_posts = agg.loc[agg['domain'] == dname, 'count'].sum()
            G.add_node(dname, type='domain', size=total_posts)

        for s in agg['subreddit'].unique():
            total_posts = agg.loc[agg['subreddit'] == s, 'count'].sum()
            G.add_node(s, type='subreddit', size=total_posts)

        for _, row in agg.iterrows():
            domain_node = row['domain']
            subreddit = row['subreddit']
            weight = int(row['count'])
            G.add_edge(domain_node, subreddit, weight=weight)

    else:
        return None, pd.DataFrame(), pd.DataFrame()

    if G.number_of_nodes() == 0:
        return None, pd.DataFrame(), pd.DataFrame()

    # Layout
    pos = nx.spring_layout(G, k=0.6, seed=42)

    # Edges
    edge_x = []
    edge_y = []
    edge_weights = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_weights.append(data.get('weight', 1))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines"
    )

    # Nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []

    for node, attrs in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        n_type = attrs.get("type", "subreddit")
        size = attrs.get("size", 1)
        scaled_size = 10 + 3 * np.log1p(size)
        node_size.append(scaled_size)

        if n_type == "subreddit":
            node_color.append("#1f77b4")  # blue
        elif n_type == "cluster":
            node_color.append("#ff7f0e")  # orange
        elif n_type == "emotion":
            node_color.append("#2ca02c")  # green
        else:
            node_color.append("#d62728")  # red (domain or others)

        node_text.append(f"{node} — {size} posts")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(n) for n in G.nodes()],
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            size=node_size,
            color=node_color,
            line_width=1
        )
    )

    title_map = {
        'subreddit_cluster': 'Narrative–Community Network (Subreddit ↔ Topic Cluster)',
        'subreddit_emotion': 'Emotion–Community Network (Subreddit ↔ Emotion)',
        'domain_subreddit': 'Domain–Community Network (Domain ↔ Subreddit)'
    }

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title_map.get(mode, "Network Graph"),
            title_x=0.5,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    # Node / Edge summaries
    node_rows = []
    for node, attrs in G.nodes(data=True):
        node_rows.append({
            "node": node,
            "type": attrs.get("type", ""),
            "posts": attrs.get("size", 0),
            "cluster_id": attrs.get("cluster_id", None),
            "emotion": attrs.get("emotion", None),
        })
    node_stats = pd.DataFrame(node_rows).sort_values("posts", ascending=False)

    edge_rows = []
    for u, v, data in G.edges(data=True):
        edge_rows.append({
            "source": u,
            "target": v,
            "count": data.get("weight", 1)
        })
    edge_stats = pd.DataFrame(edge_rows).sort_values("count", ascending=False)

    return fig, node_stats, edge_stats

# ==========================================
# 4. UI LAYOUT
# ==========================================

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    gemini_key = st.text_input("Gemini API Key", type="password", help="Required for AI Insights and Chatbot")
    
    st.divider()
    
    st.header("📂 Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
   
    
    st.divider()
    

# --- Main App ---
st.title("Reddit Domain Intel AI")
st.markdown("Analyze news domains spreading on Reddit using Sentiment Analysis, Emotion AI, Networks, and LLMs.")

# Load Resources
embedder, sentiment_model, toxicity_pipe, emotion_pipe = load_nlp_models()

# Load Data
if uploaded_file:
    raw_df = load_data(uploaded_file)
# elif use_sample:
#     raw_df = generate_dummy_data()
#     if 'sentiment' not in raw_df.columns:
#         raw_df = compute_features(raw_df, sentiment_model, None, None)
#         raw_df['toxicity'] = np.random.uniform(0, 0.1, len(raw_df))
#         raw_df['emotion'] = np.random.choice(['neutral', 'anger', 'joy', 'surprise'], len(raw_df))
else:
    st.warning("Please upload a CSV file.")
    st.stop()

# Ensure we have data
if raw_df is not None:
    # --- Feature Engineering Trigger ---
    if 'emotion' not in raw_df.columns:
        st.warning("Dataset missing computed columns (sentiment, toxicity, emotion). Calculation may take time.")
        if st.button(" Calculate AI Features Now"):
            df = compute_features(raw_df, sentiment_model, toxicity_pipe, emotion_pipe)
            st.session_state['processed_df'] = df
            st.rerun()
        else:
            df = raw_df.copy()
            if 'sentiment' not in df.columns: df['sentiment'] = 0
            if 'toxicity' not in df.columns: df['toxicity'] = 0
            if 'emotion' not in df.columns: df['emotion'] = 'neutral'
    else:
        df = raw_df

    # Precompute list of domains for use in multiple tabs
    all_domains = df['domain'].dropna().unique().tolist()
    if len(all_domains) == 0:
        st.error("No 'domain' column or it is empty in the dataset.")
        st.stop()

    default_idx = all_domains.index('youtube.com') if 'youtube.com' in all_domains else 0

    # --- Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Domain Analysis", "⚔️ Comparison", "💬 AI Chatbot", "🔗 Network Explorer"])

    # ---------------------------------------------------------
    # TAB 1: SINGLE DOMAIN ANALYSIS
    # ---------------------------------------------------------
    with tab1:
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_domain = st.selectbox("Select Domain to Analyze", options=all_domains, index=default_idx)
        
        d_df = df[df['domain'].str.contains(selected_domain, case=False, na=False)].copy()
        
        if d_df.empty:
            st.warning("No data found for this domain.")
        else:
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Posts", len(d_df))
            m2.metric("Avg Sentiment", f"{d_df['sentiment'].mean():.2f}")
            m3.metric("Avg Toxicity", f"{d_df['toxicity'].mean():.3f}")
            top_emo = d_df['emotion'].value_counts().idxmax()
            m4.metric("Dominant Emotion", top_emo.title())

            # Charts Row 1
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("Emotion Distribution")
                emo_counts = d_df['emotion'].value_counts().reset_index()
                emo_counts.columns = ['emotion', 'count']
                fig_emo = px.bar(
                    emo_counts, x='emotion', y='count', color='emotion',
                    title="Emotional Tone of Headlines", template="plotly_white"
                )
                st.plotly_chart(fig_emo, use_container_width=True)
            
            with c2:
                st.subheader("Top Subreddits")
                sub_counts = d_df['subreddit'].value_counts().head(10).reset_index()
                sub_counts.columns = ['subreddit', 'count']
                fig_sub = px.bar(
                    sub_counts, x='count', y='subreddit', orientation='h',
                    title=f"Where is {selected_domain} being shared?", template="plotly_white"
                )
                fig_sub.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_sub, use_container_width=True)

            # Timeline
            st.subheader("Posting Activity Over Time")
            if 'created_utc' in d_df.columns:
                daily_counts = d_df.set_index('created_utc').resample('D').size().reset_index(name='count')
                fig_time = px.line(
                    daily_counts, x='created_utc', y='count', markers=True,
                    title="Daily Discussion Volume", template="plotly_white"
                )
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("No 'created_utc' column found for timeline plot.")

            # AI Insight Section
            st.markdown("### Insights")
            if st.button("Generate Analytical Report"):
                if gemini_key:
                    with st.spinner("Gemini is analyzing the data..."):
                        d_df_clustered = perform_clustering(d_df)
                        num_clusters = d_df_clustered['cluster'].nunique() if 'cluster' in d_df_clustered.columns else 0
                        top_sub = d_df['subreddit'].value_counts().idxmax()
                        
                        prompt = f"""
                        Analyze these Reddit engagement metrics for the domain '{selected_domain}':
                        - Total Posts: {len(d_df)}
                        - Sentiment Score: {d_df['sentiment'].mean():.2f} (-1 to 1)
                        - Toxicity Score: {d_df['toxicity'].mean():.2f} (0 to 1)
                        - Dominant Emotion: {top_emo}
                        - Top Subreddit: r/{top_sub}
                        - Thematic Clusters found: {num_clusters}
                        
                        Provide a 4-bullet point executive summary on how this domain is being perceived and circulated.
                        """
                        insight = get_gemini_response(prompt, gemini_key)
                        st.success("Analysis Complete")
                        st.markdown(insight)
                else:
                    st.error("Please provide a Gemini API Key in the sidebar.")

    # ---------------------------------------------------------
    # TAB 2: COMPARISON
    # ---------------------------------------------------------
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            dom1 = st.selectbox("Domain A", options=all_domains, index=0, key="comp1")
        with col2:
            dom2_options = [d for d in all_domains if d != dom1]
            dom2 = st.selectbox("Domain B", options=dom2_options, index=0 if dom2_options else 0, key="comp2")

        if dom1 and dom2:
            df1 = df[df['domain'].str.contains(dom1, case=False, na=False)]
            df2 = df[df['domain'].str.contains(dom2, case=False, na=False)]

            comp_data = {
                "Metric": ["Posts", "Sentiment", "Toxicity"],
                dom1: [len(df1), f"{df1['sentiment'].mean():.2f}", f"{df1['toxicity'].mean():.3f}"],
                dom2: [len(df2), f"{df2['sentiment'].mean():.2f}", f"{df2['toxicity'].mean():.3f}"]
            }
            st.table(pd.DataFrame(comp_data))

            if 'created_utc' in df.columns:
                st.subheader("Activity Comparison")
                ts1 = df1.set_index('created_utc').resample('D').size().reset_index(name='count')
                ts1['Domain'] = dom1
                ts2 = df2.set_index('created_utc').resample('D').size().reset_index(name='count')
                ts2['Domain'] = dom2
                
                combined_ts = pd.concat([ts1, ts2])
                fig_comp = px.line(
                    combined_ts, x='created_utc', y='count', color='Domain',
                    title=f"{dom1} vs {dom2} Volume", template="plotly_white"
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("No 'created_utc' column found for timeline comparison.")

    # ---------------------------------------------------------
    # TAB 3: AI CHATBOT (RAG)
    # ---------------------------------------------------------
    with tab3:
        # Use the same selected_domain as Tab1 default for context
        st.subheader("💬 Chat with your Dataset")
        st.markdown("Ask questions about the posts related to the **currently active domain** from Tab 1 (internally tracked).")

        # Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Ensure we have a domain context (fall back gracefully)
        domain_for_chat = 'youtube.com' if 'youtube.com' in all_domains else all_domains[0]

        # Vector Database Setup (FAISS) - Run only on domain change
        if 'faiss_index' not in st.session_state or st.session_state.get('current_domain_rag') != domain_for_chat:
            with st.spinner(f"Indexing posts for {domain_for_chat}..."):
                rag_df = df[df['domain'].str.contains(domain_for_chat, case=False, na=False)].copy()
                
                if not rag_df.empty:
                    titles = rag_df['title'].astype(str).tolist()
                    vectors = embedder.encode(titles)
                    index = faiss.IndexFlatL2(vectors.shape[1])
                    index.add(vectors)
                    
                    st.session_state['faiss_index'] = index
                    st.session_state['rag_titles'] = titles
                    st.session_state['rag_df'] = rag_df
                    st.session_state['current_domain_rag'] = domain_for_chat
                else:
                    st.session_state['faiss_index'] = None

        if prompt := st.chat_input("Ex: Why are people angry about this?"):
            if st.session_state['faiss_index'] is None:
                st.error("No data available to chat about.")
                st.stop()

            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # RAG Logic
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Analyzing patterns..."):
                    # Retrieve posts via FAISS
                    q_vec = embedder.encode([prompt])
                    D, I = st.session_state['faiss_index'].search(q_vec, k=5)
                    retrieved_posts = [st.session_state['rag_titles'][i] for i in I[0]]

                    subset = st.session_state['rag_df'][st.session_state['rag_df']['title'].isin(retrieved_posts)].copy()

                    avg_sent = subset['sentiment'].mean()
                    avg_toxic = subset['toxicity'].mean()
                    top_emotion = subset['emotion'].value_counts().idxmax()
                    top_sub = subset['subreddit'].value_counts().idxmax()
                    topics = subset['cluster'].value_counts().idxmax() if "cluster" in subset else "Unknown"

                    tone = (
                        "negative" if avg_sent < -0.1
                        else "mixed" if avg_sent < 0.1
                        else "positive"
                    )

                    rag_prompt = f"""
                    You are an expert misinformation researcher analyzing how this domain is discussed on Reddit.

                    USER QUESTION:
                    {prompt}

                    RELEVANT TITLES (context only):
                    {retrieved_posts}

                    DATA SIGNALS FROM THESE POSTS:
                    - Retrieved posts: {len(subset)}
                    - Avg sentiment tone: {tone} ({avg_sent:.2f})
                    - Avg toxicity: {avg_toxic:.2f}
                    - Dominant emotion: {top_emotion}
                    - Most active subreddit: r/{top_sub}
                    - Most common topic cluster: {topics}

                    RULES:
                    - Base your answer ONLY on these signals and titles.
                    - Write in an analytical tone (not conversational).
                    - Do NOT repeat titles.
                    - Response must be 5–6 sentences.
                    - End with one useful follow-up question for the user.
                    """

                    ai_response = get_gemini_response(rag_prompt, gemini_key)

                    full_response = (
                        f"{ai_response}\n\n"
                        f"**Top Relevant Posts:**\n"
                        + "\n".join([f"- *{p}*" for p in retrieved_posts[:3]])
                    )

                    message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

    # ---------------------------------------------------------
    # TAB 4: NETWORK EXPLORER (NEW)
    # ---------------------------------------------------------
    with tab4:
        st.subheader("🔗 Network Explorer")
        st.markdown(
            "Explore how **communities and narratives connect**.\n\n"
            "- *Subreddit ↔ Topic Cluster*: which communities push which narratives for a domain.\n"
            "- *Subreddit ↔ Emotion*: which emotions dominate in which communities.\n"
            "- *Domain ↔ Subreddit (Global)*: which domains are central in which communities."
        )

        # Mode selection
        mode_display_to_key = {
            "Subreddit ↔ Narrative Cluster (per selected domain)": "subreddit_cluster",
            "Subreddit ↔ Emotion (per selected domain)": "subreddit_emotion",
            "Domain ↔ Subreddit (global)": "domain_subreddit",
        }
        mode_display = st.selectbox(
            "Network Type",
            options=list(mode_display_to_key.keys())
        )
        mode_key = mode_display_to_key[mode_display]

        # Domain context for per-domain modes
        if mode_key in ['subreddit_cluster', 'subreddit_emotion']:
            domain_for_network = st.selectbox(
                "Domain context for this network",
                options=all_domains,
                index=default_idx,
                help="The network will only use posts for this domain."
            )
        else:
            domain_for_network = None

        # Edge weight threshold
        min_edge_weight = st.slider(
            "Minimum number of posts to draw an edge (relationship strength)",
            min_value=1, max_value=10, value=1, step=1
        )

        # Build network
        with st.spinner("Building network graph..."):
            fig_net, node_stats, edge_stats = build_network_graph(
                df=df,
                mode=mode_key,
                selected_domain=domain_for_network,
                min_edge_weight=min_edge_weight
            )

        if fig_net is None:
            st.info("Not enough data to build a meaningful network with the current settings.")
        else:
            st.plotly_chart(fig_net, use_container_width=True)

            st.caption(
                "Node colors: blue = subreddits, orange = topic clusters, "
                "green = emotions, red = domains."
            )

            # Summaries
            col_nodes, col_edges = st.columns(2)
            with col_nodes:
                st.markdown("**Node Summary**")
                st.dataframe(node_stats, use_container_width=True, height=300)
            with col_edges:
                st.markdown("**Edge Summary**")
                st.dataframe(edge_stats, use_container_width=True, height=300)

            # Drill-down: select a node and view posts
            st.markdown("### 🔍 Drill Down into a Node")
            if not node_stats.empty:
                selected_node_for_drill = st.selectbox(
                    "Select a node to inspect related posts",
                    options=node_stats['node'].tolist()
                )

                if selected_node_for_drill:
                    subset_posts = df.copy()

                    if mode_key == 'domain_subreddit':
                        if selected_node_for_drill in subset_posts['subreddit'].unique():
                            subset_posts = subset_posts[subset_posts['subreddit'] == selected_node_for_drill]
                        elif selected_node_for_drill in subset_posts['domain'].unique():
                            subset_posts = subset_posts[subset_posts['domain'] == selected_node_for_drill]
                        else:
                            subset_posts = subset_posts.head(0)

                    elif mode_key == 'subreddit_cluster':
                        subset_posts = subset_posts[subset_posts['domain'].str.contains(domain_for_network, case=False, na=False)]
                        subset_posts = perform_clustering(subset_posts)
                        if selected_node_for_drill.startswith("Cluster "):
                            cid = int(selected_node_for_drill.split(" ")[1])
                            subset_posts = subset_posts[subset_posts['cluster'] == cid]
                        else:
                            subset_posts = subset_posts[subset_posts['subreddit'] == selected_node_for_drill]

                    elif mode_key == 'subreddit_emotion':
                        subset_posts = subset_posts[subset_posts['domain'].str.contains(domain_for_network, case=False, na=False)]
                        if selected_node_for_drill.startswith("Emotion: "):
                            emo = selected_node_for_drill.split("Emotion: ", 1)[1]
                            subset_posts = subset_posts[subset_posts['emotion'] == emo]
                        else:
                            subset_posts = subset_posts[subset_posts['subreddit'] == selected_node_for_drill]

                    st.markdown(
                        f"Showing up to 50 posts related to **{selected_node_for_drill}** "
                        f"({len(subset_posts)} total posts)."
                    )
                    if not subset_posts.empty:
                        cols_to_show = [c for c in ['created_utc', 'subreddit', 'domain', 'emotion', 'title'] if c in subset_posts.columns]
                        st.dataframe(subset_posts[cols_to_show].head(50), use_container_width=True)
                    else:
                        st.info("No posts found for this node with the current filters.")
