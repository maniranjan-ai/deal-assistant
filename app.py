import streamlit as st
from login import login

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def main():
    if not st.session_state['logged_in']:
        login()
    else:
        if st.sidebar.button("Logout", key='main_page_logout'):
            st.session_state['logged_in'] = False
            st.experimental_rerun()
        st.sidebar.title("Pages")
        page = st.sidebar.selectbox("Select a page", ["Home", "RFP Summarizer", "Competition Insights",
                                                      "Contract Generation", "Pricing Retrieval & Approval",
                                                      "Proposal Drafting", "Review Mechanism", "Solution Recommendation"])

        if page == "Home":
            st.title("Home")
            st.write("Welcome to the home page!")
        elif page == "RFP Summarizer":
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


if __name__ == "__main__":
    main()
