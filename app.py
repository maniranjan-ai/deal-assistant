import streamlit as st
import login



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

    # if not st.session_state['logged_in']:
    #     login()
    # else:
    if st.sidebar.button("Logout", key='main_page_logout'):
        st.session_state['logged_in'] = False
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
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if st.session_state['logged_in']:
        main()
    else:
        login.login()
    # main()
