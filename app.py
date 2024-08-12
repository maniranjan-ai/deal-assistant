import streamlit as st
from utils.rfp_helper import *
from streamlit_cookies_manager import EncryptedCookieManager


cookies = EncryptedCookieManager(
    prefix="da_app",  # You can change the prefix to any string
    password=da_pass
)

# Load cookies (this is necessary to use cookies)
if not cookies.ready():
    st.stop()

def login():
    logo_url = "static/ACN.svg"  # Replace with the URL or path to your logo
    st.image(logo_url, width=50, use_column_width=False)

    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "password":
            st.session_state['logged_in'] = True
            cookies['logged_in'] = 'true'
            cookies.save()
            st.rerun()
        else:
            st.error("Invalid username or password")

    add_footer()


def add_footer():
    footer_html = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>&copy; 2024 Accenture. All rights reserved.</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


def main():
    logo_url = "static/ACN.svg"  # Replace with the URL or path to your logo
    st.image(logo_url, width=50, use_column_width=False)

    if st.sidebar.button("Logout", key='main_page_logout'):
        st.session_state['logged_in'] = False
        cookies['logged_in'] = 'false'
        cookies.save()
        st.experimental_rerun()
    st.sidebar.title("Select a module")
    page = st.sidebar.selectbox("", ["Home", "RFP Summarization", "Competition Insights",
                                                  "Contract Generation", "Pricing Retrieval & Approval",
                                                  "Proposal Drafting", "Review Mechanism", "Solution Recommendation"])

    if page == "Home":
        st.title("Home")
        st.write("Welcome to the home page!")
    elif page == "RFP Summarization":
        # Import and render page1.py
        import my_pages.RFP_Summarizer
        my_pages.RFP_Summarizer.render()
    elif page == "Competition Insights":
        st.title("Competition Insights")
        st.write("Welcome to Competition Insights!")
        # Import and render page2.py
        import my_pages.Competition_Insights
        my_pages.Competition_Insights.render()
    elif page == "Contract Generation":
        st.title("Contract Generation")
        st.write("Welcome to Contract Generation!")
        # Import and render page2.py
        import my_pages.Contract_Generation
        my_pages.Contract_Generation.render()
    elif page == "Pricing Retrieval & Approval":
        st.title("Pricing Retrieval & Approval")
        st.write("Welcome to Pricing Retrieval & Approval!")
        # Import and render page2.py
        import my_pages.Contract_Generation
        my_pages.Contract_Generation.render()
    elif page == "Proposal Drafting":
        st.title("Proposal Drafting")
        st.write("Welcome to Proposal Drafting!")
        # Import and render page2.py
        import my_pages.Proposal_Drafting
        my_pages.Proposal_Drafting.render()
    elif page == "Review Mechanism":
        st.title("Review Mechanism")
        st.write("Welcome to Review Mechanism!")
        # Import and render page2.py
        import my_pages.Review_Mechanism
        my_pages.Review_Mechanism.render()
    elif page == "Solution Recommendation":
        st.title("Solution Recommendation")
        st.write("Welcome to Solution Recommendation!")
        # Import and render page2.py
        import my_pages.Solution_Recommendation
        my_pages.Solution_Recommendation.render()

    add_footer()


if __name__ == "__main__":
    # Initialize session state for login
    print("Now session state is", st.session_state)

    # Initialize session state for login
    if 'logged_in' not in st.session_state:
        if cookies.get('logged_in') == 'true':
            st.session_state.logged_in = True
        else:
            st.session_state.logged_in = False
    if st.session_state.logged_in:
        main()
    else:
        login()
    # main()
