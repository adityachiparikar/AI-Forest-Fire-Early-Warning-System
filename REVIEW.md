app.py:L232-274: yagni: Four nearly identical loader functions. Replace with single generic loader function.
app.py:L331-340: shrink: Repetitive layer status display. Replace with loop over layers.
config.py:L19-56: shrink: Redundant PRESET_REGIONS structure. Consider generating similar entries programmatically.
app.py:L46-226: native: Extensive custom CSS. Consider Streamlit-native components where possible (st.alert, st.metric, etc.).
app.py:L66-67: stdlib: sys.path modification. Consider proper package structure with setup.py/pyproject.toml.
app.py:L343-345: shrink: Fusion weights display. Consider st.dataframe or st.table for cleaner presentation.
app.py:L689: delete: IEEE 7000 Ethical Rationale expander. Single smoke test/assert is ponytail minimum unless compliance requires.
app.py:L692: delete: Footer caption. Consider if essential for core functionality.

net: -42 lines possible.