import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
from io import StringIO

# --- 1. Page Configuration and Creative Title ---
st.set_page_config(
    page_title="India Votes 2024: Advanced Election Insights",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("India Votes 2024: The Grand Election Saga! 🗳️📊")
st.markdown(
    """
    Welcome to an interactive journey through the **2024 Indian General Election results!** This dashboard brings complex data to life, allowing you to explore key insights, 
    understand the numbers behind the mandate, and even dabble in predicting outcomes.
    
    _Let's dive into the democratic heartbeat of India!_
    """
)

st.markdown("---") # Separator for visual appeal

# --- Data Loading (Simulated for this example) ---
# In a real scenario, you would load your actual CSV/Excel file here.
# For demonstration, we'll create a synthetic dataset resembling election data.

@st.cache_data # Cache data to avoid reloading on every rerun
def load_and_prepare_data():
    states = ['Uttar Pradesh', 'Maharashtra', 'West Bengal', 'Bihar', 'Tamil Nadu', 
              'Karnataka', 'Rajasthan', 'Madhya Pradesh', 'Gujarat', 'Andhra Pradesh', 
              'Odisha', 'Kerala', 'Punjab', 'Haryana', 'Chhattisgarh', 'Jharkhand', 'Other States']
    
    parties = ['BJP', 'INC', 'SP', 'TMC', 'DMK', 'Shiv Sena', 'NCP', 'BJD', 'AAP', 'Other']

    data = {
        'Constituency': [f'C_{i:03d}' for i in range(1, 544)],
        # --- MODIFIED LINE: Removed 'p=' argument for simplicity and to bypass error ---
        'State': np.random.choice(states, 543), 
        'Winning_Party': np.random.choice(parties, 543, p=[0.40, 0.22, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.08]), 
        'Total_Votes_Polled': np.random.randint(500000, 1500000, 543),
        'Winning_Candidate_Votes': np.random.randint(250000, 1000000, 543),
        'RunnerUp_Candidate_Votes': np.random.randint(100000, 500000, 543),
        'Voter_Turnout_Percentage': np.random.uniform(50, 85, 543).round(2),
        'Previous_Election_Margin_Avg': np.random.uniform(5, 25, 543).round(2), # Hypothetical past data
        'Candidate_Sentiment_Score': np.random.uniform(0.3, 0.9, 543).round(2), # Hypothetical ML feature
        'Development_Index': np.random.uniform(0.1, 1.0, 543).round(2) # Hypothetical ML feature
    }
    df = pd.DataFrame(data)
    
    # Calculate derived features
    df['Winning_Margin'] = df['Winning_Candidate_Votes'] - df['RunnerUp_Candidate_Votes']
    df['Vote_Share_Winning_Party'] = (df['Winning_Candidate_Votes'] / df['Total_Votes_Polled'] * 100).round(2)

    return df

df = load_and_prepare_data()

# --- Sidebar Navigation ---
st.sidebar.header("Explore the Election Data")
page_selection = st.sidebar.radio(
    "Go to:",
    [
        "📊 Dashboard Overview", 
        "🌍 Geographical & State Trends",
        "📈 Statistical Insights", 
        "🧹 Data Deep Dive & Cleaning", 
        "🤖 Predict the Winner!"
    ]
)

# --- 2. Dashboard Overview ---
if page_selection == "📊 Dashboard Overview":
    st.header("Dashboard Overview: The Big Picture 🗺️")
    st.markdown(
        """
        Get a quick snapshot of the 2024 election results. See the overall seat distribution, 
        top parties, and key metrics at a glance.
        """
    )

    # Display key metrics
    col1, col2, col3 = st.columns(3)
    total_constituencies = len(df)
    total_votes_polled = df['Total_Votes_Polled'].sum()
    avg_voter_turnout = df['Voter_Turnout_Percentage'].mean().round(2)

    col1.metric("Total Constituencies", f"{total_constituencies} Seats")
    col2.metric("Total Votes Polled", f"{total_votes_polled/10**7:.2f} Cr") # Display in Crores (10^7)
    col3.metric("Avg. Voter Turnout", f"{avg_voter_turnout}%")

    st.markdown("---")

    st.subheader("Party-wise Seat Distribution")
    party_counts = df['Winning_Party'].value_counts().reset_index()
    party_counts.columns = ['Party', 'Seats Won']
    
    fig_seats = px.bar(
        party_counts,
        x='Party',
        y='Seats Won',
        title='Seats Won by Each Party',
        color='Party',
        template='plotly_white',
        labels={'Party': 'Political Party', 'Seats Won': 'Number of Seats'},
        text='Seats Won' # Show values on bars
    )
    fig_seats.update_traces(textposition='outside')
    st.plotly_chart(fig_seats, use_container_width=True)
    st.info("Inference: Observe the dominant parties and the distribution of seats across the political landscape.")

    st.markdown("---")

    st.subheader("Overall Vote Share by Winning Party")
    party_vote_share = df.groupby('Winning_Party')['Total_Votes_Polled'].sum().reset_index()
    party_vote_share.columns = ['Party', 'Total_Votes_Polled']
    party_vote_share['Vote_Share_Percentage'] = (party_vote_share['Total_Votes_Polled'] / party_vote_share['Total_Votes_Polled'].sum() * 100).round(2)

    fig_vote_share = px.pie(
        party_vote_share,
        names='Party',
        values='Vote_Share_Percentage',
        title='Overall Vote Share by Winning Party',
        hole=0.4, # Donut chart
        template='plotly_white'
    )
    st.plotly_chart(fig_vote_share, use_container_width=True)
    st.info("Inference: Vote share gives a different perspective than seat share, often showing a broader support base for parties.")


# --- 3. Geographical & State Trends ---
elif page_selection == "🌍 Geographical & State Trends":
    st.header("Geographical & State Trends 🗺️")
    st.markdown(
        """
        Explore election dynamics across different states of India. Identify regional strongholds, 
        voter engagement, and competitive landscapes.
        """
    )

    st.subheader("Voter Turnout: State-wise Comparison")
    state_turnout = df.groupby('State')['Voter_Turnout_Percentage'].mean().sort_values(ascending=False).reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Top 10 States by Average Voter Turnout")
        fig_top_turnout = px.bar(
            state_turnout.head(10),
            x='State',
            y='Voter_Turnout_Percentage',
            title='Top 10 States by Avg. Voter Turnout',
            color='Voter_Turnout_Percentage',
            template='plotly_white',
            labels={'Voter_Turnout_Percentage': 'Avg. Turnout (%)'}
        )
        st.plotly_chart(fig_top_turnout, use_container_width=True)
        st.info("Inference: States with consistently high turnout often indicate strong voter engagement or competitive contests.")
    
    with col2:
        st.markdown("##### Bottom 10 States by Average Voter Turnout")
        fig_bottom_turnout = px.bar(
            state_turnout.tail(10).sort_values(by='Voter_Turnout_Percentage'), # Re-sort for ascending display
            x='State',
            y='Voter_Turnout_Percentage',
            title='Bottom 10 States by Avg. Voter Turnout',
            color='Voter_Turnout_Percentage',
            template='plotly_white',
            labels={'Voter_Turnout_Percentage': 'Avg. Turnout (%)'}
        )
        st.plotly_chart(fig_bottom_turnout, use_container_width=True)
        st.info("Inference: Lower turnout might suggest voter apathy, logistical challenges, or one-sided contests.")

    st.markdown("---")

    st.subheader("Total Votes Polled: State-wise Distribution")
    state_total_votes = df.groupby('State')['Total_Votes_Polled'].sum().sort_values(ascending=False).reset_index()
    
    fig_total_votes_state = px.bar(
        state_total_votes.head(15), # Show top 15 states by total votes
        x='State',
        y='Total_Votes_Polled',
        title='Top States by Total Votes Polled (in Crores)',
        color='Total_Votes_Polled',
        template='plotly_white',
        labels={'Total_Votes_Polled': 'Total Votes Polled (Crores)'},
        text=(state_total_votes['Total_Votes_Polled'].head(15) / 10**7).round(2).astype(str) + ' Cr'
    )
    fig_total_votes_state.update_traces(textposition='outside')
    st.plotly_chart(fig_total_votes_state, use_container_width=True)
    st.info("Inference: States with higher populations naturally contribute more to the overall vote count. This highlights electoral significance.")

    st.markdown("---")

    st.subheader("Party Domination within States")
    selected_state_for_domination = st.selectbox(
        "Select a State to view Party Domination:",
        options=sorted(df['State'].unique()),
        key='state_domination_select'
    )
    state_df_dom = df[df['State'] == selected_state_for_domination]
    party_state_counts_dom = state_df_dom['Winning_Party'].value_counts().reset_index()
    party_state_counts_dom.columns = ['Party', 'Seats Won']

    fig_state_party_dom = px.pie(
        party_state_counts_dom,
        names='Party',
        values='Seats Won',
        title=f'Party-wise Seat Distribution in {selected_state_for_domination}',
        hole=0.3,
        template='plotly_white'
    )
    st.plotly_chart(fig_state_party_dom, use_container_width=True)
    st.info(f"Inference: This chart reveals the political landscape and relative strengths of parties specifically within {selected_state_for_domination}.")

    st.markdown("---")

    st.subheader("Average Winning Margin by State")
    state_avg_margin = df.groupby('State')['Winning_Margin'].mean().sort_values(ascending=False).reset_index()
    fig_avg_margin_state = px.bar(
        state_avg_margin.head(15),
        x='State',
        y='Winning_Margin',
        title='Top 15 States by Average Winning Margin',
        color='Winning_Margin',
        template='plotly_white',
        labels={'Winning_Margin': 'Average Winning Margin (Votes)'}
    )
    st.plotly_chart(fig_avg_margin_state, use_container_width=True)
    st.info("Inference: States with lower average winning margins likely experienced more closely contested elections.")


# --- 4. Statistical Insights ---
elif page_selection == "📈 Statistical Insights":
    st.header("Statistical Insights: Decoding the Mandate 🧠")
    st.markdown(
        """
        Dive into the numbers! Discover key statistical features and interactive visualizations 
        that reveal patterns and anomalies in the election results.
        """
    )

    st.subheader("Descriptive Statistics of Key Election Metrics")
    st.dataframe(df[['Total_Votes_Polled', 'Winning_Candidate_Votes', 
                     'RunnerUp_Candidate_Votes', 'Voter_Turnout_Percentage', 
                     'Winning_Margin', 'Vote_Share_Winning_Party']].describe().T)
    st.info("Inference: This table provides a quick summary of the central tendency, spread, and range of our key numerical features.")

    st.markdown("---")

    st.subheader("Voter Turnout & Winning Margin Relationship")
    fig_scatter = px.scatter(
        df,
        x='Voter_Turnout_Percentage',
        y='Winning_Margin',
        color='Winning_Party',
        hover_name='Constituency',
        title='Winning Margin vs. Voter Turnout by Winning Party',
        labels={'Voter_Turnout_Percentage': 'Voter Turnout (%)', 'Winning_Margin': 'Winning Margin (Votes)'},
        template='plotly_white'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.info("Inference: This plot helps us visually inspect if there's any correlation between how many people vote and how close the election was. Look for clusters or patterns.")

    st.markdown("---")

    st.subheader("Distribution of Voter Turnout & Winning Margins")
    col1, col2 = st.columns(2)
    with col1:
        fig_turnout = px.histogram(
            df,
            x='Voter_Turnout_Percentage',
            nbins=20,
            title='Distribution of Voter Turnout Across Constituencies',
            labels={'Voter_Turnout_Percentage': 'Voter Turnout (%)'},
            template='plotly_white'
        )
        st.plotly_chart(fig_turnout, use_container_width=True)
        st.info("Inference: The shape of this histogram indicates the most common turnout ranges and identifies any unusual peaks or troughs.")

    with col2:
        fig_margin = px.histogram(
            df,
            x='Winning_Margin',
            nbins=30,
            title='Distribution of Winning Margins',
            labels={'Winning_Margin': 'Vote Difference (Winner vs. Runner-up)'},
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template='plotly_white'
        )
        st.plotly_chart(fig_margin, use_container_width=True)
        st.info("Inference: A large peak near zero suggests many closely contested seats, while a spread indicates more landslide victories.")

    st.markdown("---")

    st.subheader("Distribution of Winning Party's Vote Share")
    fig_win_share = px.histogram(
        df,
        x='Vote_Share_Winning_Party',
        nbins=20,
        title='Distribution of Vote Share for Winning Parties',
        labels={'Vote_Share_Winning_Party': 'Winning Party\'s Vote Share (%)'},
        color_discrete_sequence=px.colors.qualitative.Dark24,
        template='plotly_white'
    )
    st.plotly_chart(fig_win_share, use_container_width=True)
    st.info("Inference: This shows how strong the mandate was for winning candidates, i.e., what percentage of votes the winner actually secured.")

    st.markdown("---")

    st.subheader("Most & Least Contested Constituencies")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 🏆 Top 10 Least Contested (Largest Winning Margin)")
        least_contested = df.nlargest(10, 'Winning_Margin')[['Constituency', 'State', 'Winning_Party', 'Winning_Margin']].round(0)
        st.dataframe(least_contested)
        st.info("Inference: These constituencies saw clear victories, often indicating strong incumbent support or less competition.")
    with col_b:
        st.markdown("##### 🥊 Top 10 Most Contested (Smallest Winning Margin)")
        most_contested = df.nsmallest(10, 'Winning_Margin')[['Constituency', 'State', 'Winning_Party', 'Winning_Margin']].round(0)
        st.dataframe(most_contested)
        st.info("Inference: These were nail-biting finishes, where even small shifts in votes could have changed the outcome.")


# --- 5. Data Deep Dive & Cleaning ---
elif page_selection == "🧹 Data Deep Dive & Cleaning":
    st.header("Data Deep Dive: Unveiling the Raw Numbers 🕵️‍♀️")
    st.markdown(
        """
        Explore the raw dataset and understand how we ensure its quality 
        through essential data cleaning steps.
        """
    )

    st.subheader("Raw Data Sample")
    st.dataframe(df.head())
    st.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    st.subheader("Initial Data Information (`df.info()`)")
    # Using io.StringIO to capture df.info() output
    buffer = StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    st.info("Inference: This output shows column names, non-null counts, and data types, essential for data quality checks.")

    st.subheader("Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        st.warning("Missing values detected! (Though simulated data aims for minimal/no missingness)")
        st.table(missing_data.sort_values(ascending=False))
        st.markdown(
            """
            *Self-correction note: In a real scenario, you'd handle these using methods 
            like `df.dropna()` (to remove rows with missing data) or `df.fillna()` (to impute/fill missing values) 
            based on the column type and context. For numerical data, mean/median imputation is common; 
            for categorical, mode imputation or a 'Unknown' category.*
            """
        )
    else:
        st.success("No significant missing values in this simulated dataset! 🎉 (Good for demonstration)")

    st.subheader("Data Type Conversion & Consistency (Demonstration)")
    st.markdown(
        """
        Ensuring data types are correct is crucial for accurate analysis and calculations. 
        For instance, ensuring vote counts are integers and percentages are floats.
        (Our simulated data is largely set, but this shows the intent.)
        """
    )
    # This section serves as a conceptual explanation, no direct changes needed for current data
    st.code("""
# Example: Convert 'Total_Votes_Polled' to integer
df['Total_Votes_Polled'] = df['Total_Votes_Polled'].astype(int)

# Example: Convert 'Voter_Turnout_Percentage' to float
df['Voter_Turnout_Percentage'] = df['Voter_Turnout_Percentage'].astype(float)
    """)
    st.info("Inference: Correct data types prevent errors in calculations and enable proper statistical analysis.")


# --- 6. Machine Learning Prediction ---
elif page_selection == "🤖 Predict the Winner!":
    st.header("Predict the Winner: The ML Crystal Ball 🔮")
    st.markdown(
        """
        Ever wondered what factors influence an election outcome? Here, we use a
        simple Machine Learning model to predict the **Winning Party** for a hypothetical constituency
        based on a few key features.
        
        **Disclaimer:** This is a simplified demonstration using synthetic data. Real-world election 
        prediction is far more complex and involves vast amounts of diverse data, sophisticated models,
        and often, external socio-economic factors.
        """
    )

    st.subheader("Model Training")
    st.write("We'll train a Decision Tree Classifier on our simulated election data. Decision Trees are intuitive and show feature importance.")
    
    # Define features (X) and target (y)
    features = ['Total_Votes_Polled', 'Voter_Turnout_Percentage', 
                'Previous_Election_Margin_Avg', 'Candidate_Sentiment_Score', 'Development_Index']
    target = 'Winning_Party'

    # Ensure X and y are defined before use in train_test_split
    X = df[features]
    y = df[target]

    # Check if target variable has enough classes for stratified split
    # Stratify requires at least 2 samples per class in each split
    target_counts = y.value_counts()
    if all(target_counts >= 2): 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        # Fallback to non-stratified split if not enough samples per class for stratification
        st.warning("Not enough samples per class for stratified split. Using non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Machine Learning Model (Decision Tree) Trained Successfully! ✨")
    st.write(f"Model Accuracy on Test Data: **{accuracy:.2f}** (This is for illustrative purposes with simulated data)")
    st.info("Inference: Model accuracy gives an idea of how well our simple model predicts winning parties on unseen data.")
    
    with st.expander("See Classification Report (Detailed Performance)"):
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
        st.info("Inference: Precision, Recall, and F1-score provide more nuanced insights into the model's performance for each party.")

    st.markdown("---")

    st.subheader("Feature Importance")
    feature_importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
    fig_feature_importance = px.bar(
        feature_importances,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance for Predicting Winning Party',
        labels={'Importance': 'Relative Importance', 'Feature': 'Input Feature'},
        template='plotly_white'
    )
    st.plotly_chart(fig_feature_importance, use_container_width=True)
    st.info("Inference: This chart shows which factors (features) the model considered most influential in predicting the winning party.")

    st.markdown("---")

    st.subheader("Predict for a Hypothetical Constituency")

    # User inputs for prediction
    st.sidebar.subheader("Adjust Features for Prediction")
    
    input_total_votes = st.sidebar.slider("Total Votes Polled:", int(df['Total_Votes_Polled'].min()), int(df['Total_Votes_Polled'].max()), int(df['Total_Votes_Polled'].mean()), key='tvp_slider')
    input_voter_turnout = st.sidebar.slider("Voter Turnout Percentage:", float(df['Voter_Turnout_Percentage'].min()), float(df['Voter_Turnout_Percentage'].max()), float(df['Voter_Turnout_Percentage'].mean()), key='vtp_slider')
    input_prev_margin = st.sidebar.slider("Previous Election Margin (Avg. %):", float(df['Previous_Election_Margin_Avg'].min()), float(df['Previous_Election_Margin_Avg'].max()), float(df['Previous_Election_Margin_Avg'].mean()), key='pem_slider')
    input_sentiment_score = st.sidebar.slider("Candidate Sentiment Score (0.0-1.0):", 0.0, 1.0, float(df['Candidate_Sentiment_Score'].mean()), 0.01, key='css_slider')
    input_development_index = st.sidebar.slider("Local Development Index (0.0-1.0):", 0.0, 1.0, float(df['Development_Index'].mean()), 0.01, key='ldi_slider')

    predict_button = st.button("Predict Winning Party")

    if predict_button:
        input_data = pd.DataFrame([[input_total_votes, input_voter_turnout, 
                                     input_prev_margin, input_sentiment_score, 
                                     input_development_index]], 
                                    columns=features)
        
        predicted_party = model.predict(input_data)[0]
        
        st.markdown(f"### 🎉 Based on your inputs, the predicted winning party is: **{predicted_party}**")
        st.balloons()

        st.markdown(
            """
            *Remember, this prediction is purely illustrative. Election outcomes are influenced by a myriad of complex, 
            often unpredictable, real-world factors beyond simple numerical inputs. This model serves as a glimpse into how 
            machine learning can be applied to electoral data.*
            """
        )

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: small; color: gray;">
        Built with ❤️ using Streamlit for Data Enthusiasts.
    </div>
    """,
    unsafe_allow_html=True
)