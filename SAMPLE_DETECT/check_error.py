import traceback

try:
    import streamlit_app
    print("SUCCESS")
except Exception as e:
    print("ERROR:")
    traceback.print_exc()
