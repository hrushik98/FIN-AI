import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import openai
import wget
import os
from yahooquery import Ticker
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from openai import OpenAI
client = OpenAI(api_key = st.secrets['OPENAI_API_KEY'])

st.title("Financial Analysis with AI")



def reset():
    if os.path.exists("balance_sheet.csv"):
        os.remove("balance_sheet.csv")
    if os.path.exists("financial_data.csv"):
        os.remove("financial_data.csv")
    if os.path.exists("income_statement.csv"):
        os.remove("income_statement.csv")
    if os.path.exists("tickers.csv"):
        os.remove("tickers.csv")



################################################################################# GETTING THE FINANCIAL STATEMENTS #################################################################################

def get_financial_statements(company_name):
    company_name = company_name.lower()
    lst = []
    tickers = pd.read_csv("tickers.csv")
    for i in range(len(tickers)):
        if company_name in tickers['Company Name'][i].lower():
            print("Ticker: ", tickers['Symbol'][i])
            lst.append(str(tickers['Symbol'][i]))
            break
    ticker_symbol = lst[0]
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period="1y")
    # Assuming hist dataframe is already loaded
    # Select 'Open' and 'Close' columns from the hist dataframe
    hist_selected = hist[['Open', 'Close']]
    # Create a new figure in Plotly
    fig = go.Figure()
    # Plot the selected data using Scatter mode (lines)
    colors = {'Open': 'blue', 'Close': 'green'}
    for col in hist_selected.columns:
        fig.add_trace(go.Scatter(x=hist_selected.index, y=hist_selected[col], mode='lines', name=col, line=dict(color=colors[col])))
    # Set the title and labels
    fig.update_layout(
        title=f"{company_name.capitalize()} Stock Price",
        xaxis=dict(title='Date'),
        yaxis=dict(title='Stock Price')
    )
    # Display the plot in Streamlit
    st.plotly_chart(fig)
    company = Ticker(ticker_symbol)
    freq = "a"   # "a" -> annual , "q" -> quarterly

    balance_sheet = company.balance_sheet(frequency=freq)
    income_statement = company.income_statement(frequency=freq)
    balance_sheet.to_csv("balance_sheet.csv")
    data = income_statement
    max_rep_date = []

    def response1(company_name):
        global search_list
        import os
        import requests
        system_content = f"""
        You are given a company name and the recent news related to it.
        Present information in markdown format to display it.
        <h1> 💡 Overview <h1>
        <p> Give a short overview of the company.</p>
        """
        company_name = company_name
        user_content = f"""company_name = {company_name} """

        m1 = [{"role": "system", "content": f"{system_content}"},
              {"role": "user", "content": f"financial_data: {user_content}"}]
        result = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.8,
            messages=m1)
        response = result.choices[0].message.content
        st.markdown(response, unsafe_allow_html=True)


##################################################################### GETTING INDUSTRY OUTLOOK, TRENDS, ETC... #################################################################################

        search_list = ['Recent Developments', 'Market Trends', 'Industry Outlook']
        for to_search in search_list:
            links = []
        # Query to search for news articles
            query = f"{to_search} {company_name}"
            # Perform the search
            search_results = search(query, num_results=5, lang="en")
            # Print the search results
            for i, result in enumerate(search_results, start=1):
                if result not in links:
                    links.append(result)
            scraped_text = ""
            for link in links:
                url = f"{link}"
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    print(f"Error accessing the website: {e}")
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                scraped_text+=text
            system_content = f"""You are a financial analyst. Generate a paragraph about the {to_search} at {company_name} company, by using the user provided infomation as reference. Don't try to make up your own information."""
            user_content = f"Here's what various news articles say about the {to_search} at {company_name}: {scraped_text} \n <h1> {query} </h1> \n//Give your answer here."
            recent_developments = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                        {
                        "role": "system",
                        "content": system_content,
                    },
                    {
                        "role": "user",
                        "content": user_content[:40000]
                    },

                ],
            )
            st.header(f"{to_search}")
            st.write(recent_developments.choices[0].message.content)

    response1(company_name)

    def max_repeating_element(lst):
        max_element = None
        max_count = 0
        for element in lst:
            count = lst.count(element)
            if count > max_count:
                max_element = element
                max_count = count
        return max_element, max_count

    data['asOfDate'] = data['asOfDate'].apply(lambda date: date.strftime('%Y-%m-%d'))
    for x in data['asOfDate']:
        max_rep_date.append(str(x[5:]))
    max_repeating_date = max_repeating_element(max_rep_date)[0]
    data = data[data['asOfDate'].str.contains(max_repeating_date)].reset_index(drop=True)[:-1]
    data.to_csv(f"income_statement.csv")
    print(f"balance_sheet.csv", " has been created")
    print(f"income_statement.csv", " has been created")

def get_search_links(company_name):
  global search_list
  for to_search in search_list:
    display_links = []
    query = f"{to_search} {company_name}"
    search_results = search(query, num_results=5, lang="en")
    st.write(f"{to_search} \n")
    for i, result in enumerate(search_results, start=1):
      display_links.append(result)
    for i in range(0,5):
      st.write(display_links[i])
    st.write("---")


################################################################################# PLOTTING THE 4 GRAPHS #################################################################################


def plot_graphs():

    #Graph 1
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    st.header("📊 Financial Performance Analysis")
    fin_data = pd.read_csv("financial_data.csv")

    narration_dates = fin_data['Date']
    sales = fin_data['Sales']
    net_op_margin = fin_data['Net Op Margin']
    sales_growth = fin_data['Sales Growth']

    # Convert dates to pandas datetime objects for better visualization on the x-axis
    narration_dates_dt = pd.to_datetime(narration_dates)

    # Remove the None value from sales_growth
    sales_growth_filtered = [val if val is not None else 0 for val in sales_growth]

    # Create subplots with two Y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot the sales data as a bar graph
    fig.add_trace(go.Bar(x=narration_dates_dt, y=sales, name='Sales', marker_color='blue'), secondary_y=False)

    # Plot the net operating margin data as a line graph
    fig.add_trace(go.Scatter(x=narration_dates_dt, y=net_op_margin, mode='markers+lines', name='Net Operating Margin', line=dict(color='red'), marker=dict(color='red', symbol='circle-open')), secondary_y=True)

    # Plot the sales growth data as a line graph
    fig.add_trace(go.Scatter(x=narration_dates_dt, y=sales_growth_filtered, mode='markers+lines', name='Sales Growth', line=dict(color='green', dash='dash'), marker=dict(color='green', symbol='square')), secondary_y=True)

    # Set the title and labels
    fig.update_layout(title='Graph 1: Sales & Margin', xaxis_title='Year')

    # Set Y-axis titles for both axes
    fig.update_yaxes(title_text="Sales", secondary_y=False)
    fig.update_yaxes(title_text="Sales Growth / Net Operating Margin", secondary_y=True)

    # Show legends
    fig.update_layout(legend=dict(y=0.9, x=1.1))

    # Show the plot
    st.plotly_chart(fig)


    # Code for Graph 2 - FCFF over time
    narration_dates_fcff = fin_data['Date']
    years_fcff = pd.to_datetime(narration_dates_fcff).dt.year
    fcff = fin_data['FCFF']

    fig_fcff = go.Figure()
    fig_fcff.add_trace(go.Bar(x=years_fcff, y=fcff, name='FCFF', marker_color='blue', opacity=0.7))
    fig_fcff.update_layout(
        title='Graph 2: FCFF over time',
        xaxis_title='Year',  # Update x-axis label
        yaxis_title='FCFF',
        legend= dict(y=0.9, x=1.1),
        width=800,
        height=500,
        showlegend=True,
        xaxis=dict(
            tickmode='linear',  # Set tick mode to linear
            dtick=1  # Set tick interval to 1
        )
    )
    st.plotly_chart(fig_fcff)

    # Code for Graph 3 - Asset Turnover and Net Op Margin
    narration_dates_asset_to = fin_data['Date']
    years = pd.to_datetime(narration_dates_asset_to).dt.year
    asset_to = fin_data['Asset TO']
    net_op_margin = fin_data['Net Op Margin']
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=years, y=asset_to, mode='markers+lines', name='Asset TO',
                              marker=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=years, y=net_op_margin, mode='markers+lines',
                              name='Net Op Margin', marker=dict(color='red')))
    fig3.update_layout(title='Graph 3: Asset Turnover and Net Op Margin',
                    xaxis_title='Year',  # Update x-axis label
                    yaxis=dict(title='Asset Turnover', titlefont=dict(color='blue'),
                                tickfont=dict(color='blue')),
                    yaxis2=dict(title='Net Op Margin', titlefont=dict(color='red'),
                                tickfont=dict(color='red'), overlaying='y', side='right'),
                    xaxis=dict(
                        tickmode='linear',  # Set tick mode to linear
                        dtick=1  # Set tick interval to 1
                    ),
                    xaxis_tickangle=0)
    st.plotly_chart(fig3)
    narration_dates_reinvestment = fin_data['Date']
    years_reinvestment = pd.to_datetime(narration_dates_reinvestment).dt.year
    reinvestment_percent = fin_data['Reinvestment %']
    roic = fin_data['ROIC']

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=years_reinvestment, y=reinvestment_percent, mode='markers+lines',
                            name='Reinvestment %', marker=dict(color='blue')))
    fig4.add_trace(go.Scatter(x=years_reinvestment, y=roic, mode='markers+lines', name='ROIC',
                            marker=dict(color='red')))
    fig4.update_layout(title='Graph 4: Reinvestment % and ROIC',
                    xaxis_title='Year',  # Update x-axis label
                    yaxis=dict(title='Reinvestment %', titlefont=dict(color='blue'),
                                tickfont=dict(color='blue')),
                    yaxis2=dict(title='ROIC', titlefont=dict(color='red'),
                                tickfont=dict(color='red'), overlaying='y', side='right'),
                    xaxis=dict(
                        tickmode='linear',  # Set tick mode to linear
                        dtick=1  # Set tick interval to 1
                    ),
                    xaxis_tickangle=0)
    st.plotly_chart(fig4)





################################################################################# CREATING FINANCIAL DATA TABLE USING FORMULAS #################################################################################


import pandas as pd
import csv

def create_financial_data():
    bs_data = pd.read_csv("balance_sheet.csv")
    is_data = pd.read_csv("income_statement.csv")
    import csv
    import csv

    # Define the CSV file name
    file_name = 'financial_data.csv'

    # Define the column headers
    headers = ['Date', 'Sales', 'Operating Profit', 'Tax Rate', 'NOPAT', 'Reinvestment', 'FCFF', 'WC', 'Fixed Asset', 'Operating Asset', 'Asset TO', 'Net Op Margin', 'ROIC', 'Reinvestment %', 'Sales Growth']

    # Write an empty row for the data (to create an empty CSV file with headers only)
    data = []

    # Write the data to the CSV file
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(headers)
        # Write the data rows (empty in this case)
        writer.writerows(data)
    try:
        for i in range(0, len(bs_data)):
            with open("financial_data.csv", "a") as f:
                datai = csv.writer(f)
                col1 = bs_data['asOfDate'][i]
                col2 = is_data['TotalRevenue'][i]
                col3 = is_data['EBIT'][i]
                col4 = is_data['TaxProvision'][i] / is_data['PretaxIncome'][i]
                col5 = col3 * (1 - col4)
                try:
                    col6 = bs_data['WorkingCapital'][i] + bs_data['NetPPE'][i] + bs_data['ConstructionInProgress'][i] - (bs_data['WorkingCapital'][i - 1] + bs_data['NetPPE'][i - 1] + bs_data['ConstructionInProgress'][i - 1])
                except:
                    try:
                        col6 = bs_data['WorkingCapital'][i] + bs_data['NetPPE'][i]  - (bs_data['WorkingCapital'][i - 1] + bs_data['NetPPE'][i - 1] )
                    except:
                        col6 = 0
                col7 = col5 - col6
                col8 = bs_data['WorkingCapital'][i]
                try:
                    col9 = bs_data['NetPPE'][i] + bs_data['ConstructionInProgress'][i]
                except:
                    col9 = bs_data['NetPPE'][i]
                col10 = col8 + col9
                try:
                    col11 = col2 * 2 / (col10 + pd.read_csv("financial_data")['Operating Asset'][i - 1])
                except:
                    col11 = col2 * 2 / (col10 * 2)
                col12 = col5 / col2
                try:
                    col13 = col5 * 2 / (col10 + pd.read_csv("financial_data")['Operating Asset'][i - 1])
                except:
                    col13 = col5 * 2 / (col10 * 2)
                col14 = col6 / abs(col5)
                try:
                    prev_narration_dates_rev = is_data['TotalRevenue'][i - 1]
                    col15 = col2 / prev_narration_dates_rev - 1
                except:
                    col15 = "Data N/A"
                datai.writerow([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15])
    except:

        for i in range(0, (len(bs_data)-1)):
            with open("financial_data.csv", "a") as f:
                datai = csv.writer(f)
                col1 = bs_data['asOfDate'][i]
                col2 = is_data['TotalRevenue'][i]
                col3 = is_data['EBIT'][i]
                col4 = is_data['TaxProvision'][i] / is_data['PretaxIncome'][i]
                col5 = col3 * (1 - col4)
                try:
                    col6 = bs_data['WorkingCapital'][i] + bs_data['NetPPE'][i] + bs_data['ConstructionInProgress'][i] - (bs_data['WorkingCapital'][i - 1] + bs_data['NetPPE'][i - 1] + bs_data['ConstructionInProgress'][i - 1])
                except:
                    try:
                        col6 = bs_data['WorkingCapital'][i] + bs_data['NetPPE'][i]  - (bs_data['WorkingCapital'][i - 1] + bs_data['NetPPE'][i - 1] )
                    except:
                        col6 = 0
                col7 = col5 - col6
                col8 = bs_data['WorkingCapital'][i]
                try:
                    col9 = bs_data['NetPPE'][i] + bs_data['ConstructionInProgress'][i]
                except:
                    col9 = bs_data['NetPPE'][i]
                col10 = col8 + col9
                try:
                    col11 = col2 * 2 / (col10 + pd.read_csv("financial_data")['Operating Asset'][i - 1])
                except:
                    col11 = col2 * 2 / (col10 * 2)
                col12 = col5 / col2
                try:
                    col13 = col5 * 2 / (col10 + pd.read_csv("financial_data")['Operating Asset'][i - 1])
                except:
                    col13 = col5 * 2 / (col10 * 2)
                col14 = col6 / abs(col5)
                try:
                    prev_narration_dates_rev = is_data['TotalRevenue'][i - 1]
                    col15 = col2 / prev_narration_dates_rev - 1
                except:
                    col15 = "Data N/A"
                datai.writerow([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15])




################################################################################# ANALYSIS AND RECOMMENDATION #################################################################################


def response2():
    pd.read_csv("financial_data.csv").transpose()
    text = pd.read_csv("financial_data.csv").transpose().to_string()
    system_content = f"""You are a financial assistant, with a vast expertise in business finance.
    You will be given the analysis of a particular company and you have to perform financial analysis

    Include these headings:
    <h2> Analysis <h2>
    //Analyse the financial performace of the company using the user provided data.
    <h2> 💰 Ivestment Recommendation<h2>
    write a detailled investment thesis to answer
    the user request as a html document.
    Provide numbers to justify your assertions, a lot ideally.
    Always provide a recommendation to buy the stock of the company
    or not, given the information available.
    Please give you response in markdown
    to display it.

    """
    m1 = [{"role": "system", "content": f"{system_content}"},
        {"role": "user", "content": f"financial_data: {text}"}]
    result = client.chat.completions.create(
    model="gpt-4o",
    temperature =0.8,
    messages=m1)
    response = result.choices[0].message.content
    st.markdown(response, unsafe_allow_html=True)


################################################################################# SOURCES #################################################################################

def view_sources():
    global search_list
    global links
    global search_list_links
    st.title("🔗 Sources")
    st.write("Balance Sheet")
    st.write(pd.read_csv("balance_sheet.csv"))
    st.text("")
    st.write("Income Statement")
    st.write(pd.read_csv("income_statement.csv"))
    st.text("")
    st.write("Financial Data")
    st.write(pd.read_csv("financial_data.csv"))
    st.header("External Sources")
    st.write("---")
    get_search_links(company_name)

############################################################################################################################################################################################################
company_name = st.text_input("Enter the company name")
if st.button("Analyze"):
    reset()
    url = "https://drive.google.com/uc?export=download&id=1BW-te6Qi6UBCobrVeBprW7jffLvNdkcf"
    output_file = "tickers.csv"
    wget.download(url, out=output_file)
    get_financial_statements(company_name)
    st.text("")
    st.text("")
    create_financial_data()
    plot_graphs()
    response2()
    st.text("")
    view_sources()


