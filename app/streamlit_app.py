import os
import sys
import pathlib
import streamlit as st

# Ensure project root is on sys.path so `import app.recommender` works when Streamlit runs this file.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.recommender import BookRecommender

st.set_page_config(page_title='LLM Book Recommender', layout='wide')

st.title('LLM Book Recommender')
st.write('Enter a short description of what you want to read and the system will recommend books from the dataset.')
@st.cache_resource
def get_recommender():
    return BookRecommender()

recommender = get_recommender()

with st.sidebar:
    st.header('Settings')
    # Choose model should be first according to user's preference
    st.subheader('Ollama (local)')
    # placeholder for model selectbox (we populate later)
    st.markdown('Choose model (installed or enter name below)')
    # model selectbox/text will be rendered after recommender is created
    k = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)
    use_ollama = st.checkbox('Enable Ollama explanations (local)', value=True)
    st.markdown('**Explanation style**')
    short = st.checkbox('Short (1-2 sentences)', value=True)
    structured = st.checkbox('Structured JSON output', value=False)
    # auto-pull option removed: automatic model pulling is no longer supported from UI
    st.markdown('---')
    st.markdown('Enter model name if not in the installed list:')
    # text input for manual model name (in case none installed)
    # place a small pull button next to the input
    cols = st.columns([3, 1])
    with cols[0]:
        manual_model = st.text_input('Manual model name (optional)', value='')
    with cols[1]:
        manual_pull_btn = st.button('Pull')
    # small status placeholder for manual pulls
    manual_pull_status = st.empty()
    # handle manual pull immediately when button clicked
    if manual_pull_btn:
        target = (manual_model or '').strip()
        if not target:
            manual_pull_status.warning('Enter a manual model name to pull')
        else:
            manual_pull_status.text(f'Attempting to pull model: {target} ...')
            try:
                ok, status = recommender._ensure_ollama_model(target)
            except Exception as e:
                ok, status = False, str(e)
            if ok:
                # refresh local models and try to canonicalize the pulled model name
                local_models = recommender._get_local_ollama_models()
                match = None
                for m in local_models:
                    if m == target or m.startswith(target) or target.startswith(m):
                        match = m
                        break
                if match:
                    if 'embed' in match.lower():
                        manual_pull_status.success(f'Model "{match}" pulled but appears to be embedding-only; not added to recommendation dropdown.')
                    else:
                        # update session so dropdown shows the pulled model
                        st.session_state['ollama_model'] = match
                        manual_pull_status.success(f'Model "{match}" pulled and added to dropdown.')
                        # rerun so UI updates (selectbox will pick the new model)
                        st.experimental_rerun()
                else:
                    manual_pull_status.success('Model pulled successfully, but it was not found in local model list. Use Refresh in health-check.')
            else:
                # handle special case where Ollama CLI is outdated
                if isinstance(status, str) and status.startswith('ollama_outdated'):
                    # extract message after prefix
                    msg = status.split(':', 1)[1] if ':' in status else status
                    manual_pull_status.error('Pull failed: your Ollama CLI is out of date. Please update Ollama to the latest version.')
                    manual_pull_status.info('Download: https://ollama.com/download')
                    # also show the sanitized CLI message
                    manual_pull_status.write(msg)
                else:
                    manual_pull_status.error(f'Pull failed: {status}')

# populate model list after recommender is available
# fetch models
models = recommender._get_local_ollama_models()
# filter out embedding models
text_models = [m for m in models if 'embed' not in m.lower()]
# drop likely embedding-only models (names that include 'embed')
text_models = [m for m in models if 'embed' not in m.lower()]
if text_models:
    # prefer a sensible default if available
    preferred = None
    for pref in ['llama3.2:latest', 'llama3.2', 'llava:7b', 'Dragon:latest']:
        for m in text_models:
            if m == pref or m.startswith(pref):
                preferred = m
                break
        if preferred:
            break
    default_index = 0
    options = [''] + text_models
    if preferred and preferred in text_models:
        default_index = options.index(preferred)
    if 'ollama_model' not in st.session_state:
        st.session_state['ollama_model'] = options[default_index]
    selected = st.selectbox('Choose Ollama model (installed)', options=options, index=default_index)
    st.session_state['ollama_model'] = selected
    ollama_model = st.session_state.get('ollama_model', '')
else:
    # prefer manual_model if provided
    ollama_model = manual_model or os.environ.get('OLLAMA_MODEL', '')

# Pull model UI: confirmation and progress
pull_placeholder = st.sidebar.empty()
if use_ollama:
    # only show model actions if the currently selected model isn't already installed
    local_models = recommender._get_local_ollama_models()
    selected_installed = False
    if ollama_model:
        for m in local_models:
            if m == ollama_model or (ollama_model and m.startswith(ollama_model)):
                selected_installed = True
                break

    # show Model actions only when there's no installed selection (i.e., pulling may be required)
    if not selected_installed:
        with st.sidebar.expander('Model actions'):
            st.write('Selected model:', ollama_model or '(none)')
            # Allow pulling either the selected installed model or a manually-entered model
            can_pull = bool((ollama_model and ollama_model.strip()) or (manual_model and manual_model.strip()))
            if can_pull:
                if st.button('Pull model'):
                    confirm = st.confirm('Pulling a model may be large. Continue?') if hasattr(st, 'confirm') else True
                    if confirm:
                        import subprocess

                        out_area = pull_placeholder
                        out_area.text('Starting pull...')
                        # decide which model to pull: prefer selected installed model, otherwise manual input
                        target = (ollama_model or manual_model or '').strip()
                        if not target:
                            out_area.text('No model specified to pull')
                        else:
                            try:
                                proc = subprocess.Popen(['ollama', 'pull', target], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                                # stream simple progress lines into the placeholder
                                for line in proc.stdout:
                                    if line.strip():
                                        out_area.text(line.strip())
                                proc.wait()
                                if proc.returncode == 0:
                                    # refresh local models and try to find a canonical match for the requested model
                                    local_models = recommender._get_local_ollama_models()
                                    match = None
                                    for m in local_models:
                                        if m == target or m.startswith(target) or target.startswith(m):
                                            match = m
                                            break
                                    if match:
                                        # check suitability: skip embedding-only models
                                        if 'embed' in match.lower():
                                            out_area.text(f'Pull completed, but model "{match}" looks like an embedding model and is not included in the recommendation dropdown.')
                                            st.success(f'Model "{match}" pulled (embedding-only).')
                                            # still rerun so health-check shows the new model
                                            st.experimental_rerun()
                                        else:
                                            # set session state so the dropdown will show the new model as selected after rerun
                                            st.session_state['ollama_model'] = match
                                            out_area.text(f'Pull completed successfully. Added model: {match}')
                                            st.success(f'Model "{match}" pulled and selected.')
                                            st.experimental_rerun()
                                    else:
                                        out_area.text('Pull completed but model not found in local list; refresh to check available models.')
                                        st.success('Pull finished; refresh the model list using the health-check panel.')
                                else:
                                    # sanitize combined output
                                    raw = (proc.stdout or '') + '\n' + (proc.stderr or '')
                                    import re
                                    clean = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', raw).strip()
                                    # detect outdated ollama error
                                    if 'requires a newer version' in clean.lower() or 'pull model manifest: 412' in clean.lower():
                                        out_area.text('Pull failed: your Ollama CLI is out of date. Please update to the latest version: https://ollama.com/download')
                                        st.error('Pull failed: outdated Ollama CLI')
                                        st.info('https://ollama.com/download')
                                    else:
                                        out_area.text(f'Pull failed (exit {proc.returncode})')
                                        st.error(f'Pull failed (exit {proc.returncode})')
                            except FileNotFoundError:
                                out_area.text('ollama CLI not found')
                                st.error('ollama CLI not found on PATH')
                            except Exception as e:
                                out_area.text(f'Error pulling model: {e}')
                                st.error(f'Error pulling model: {e}')

query = st.text_area('Describe your interests or ask for a recommendation', height=120)

if st.button('Recommend'):
    if not query.strip():
        st.warning('Please enter a query')
    else:
        with st.spinner('Finding books...'):
            recs = recommender.recommend(query, k=k)
            if use_ollama:
                with st.spinner('Generating Ollama explanations...'):
                    recs = recommender.explain_with_ollama(
                        recs,
                        query,
                        model=ollama_model or None,
                        auto_pull=False,
                        structured=structured,
                        short=short,
                    )
        for r in recs:
            st.markdown(f"### {r['title']} — {r['authors']} ({r['published_year']})")
            if r['subtitle']:
                st.markdown(f"*{r['subtitle']}*")
            st.write(r['description'][:800] + ('...' if len(r['description'])>800 else ''))
            st.write(f"**Match score:** {r['score']:.4f}")
            if 'explanation' in r and r['explanation']:
                st.markdown(f"**Ollama explanation:** {r['explanation']}")
            if 'explanation_json' in r and r['explanation_json']:
                st.markdown('**Structured explanation (parsed)**')
                ej = r['explanation_json']
                # display reason, themes, confidence if present
                if isinstance(ej, dict):
                    if 'reason' in ej:
                        st.write('Reason:', ej.get('reason'))
                    if 'themes' in ej:
                        st.write('Themes:', ', '.join(ej.get('themes') or []))
                    if 'confidence' in ej:
                        st.write('Confidence:', ej.get('confidence'))
                else:
                    st.write(ej)
            st.write('---')

with st.sidebar.expander('Ollama health-check', expanded=False):
    st.write('Local Ollama models installed:')
    models = recommender._get_local_ollama_models()
    if models:
        for m in models:
            st.write('-', m)
    else:
        st.write('No local models detected. Run `ollama pull <model>` or enable auto-pull.')
    if st.button('Refresh model list'):
        st.experimental_rerun()

st.caption('Embeddings computed with sentence-transformers; no external LLM calls required for basic recommendations. Enable "Ollama explanations (local)" to use a local Ollama model (requires `ollama` CLI and a local model).')
