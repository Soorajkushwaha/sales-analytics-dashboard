import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import numpy as np
import warnings

# Streamlit page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Enhanced CSS for better styling
st.markdown("""
<style>
    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
    }

    /* Metric containers */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #4e73df 0%, #224abe 100%);
        color: white;
    }

    /* Sidebar filters */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .stMultiSelect > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Date input styling */
    .stDateInput > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        border: 2px solid #ffffff20;
        padding: 8px 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .stDateInput > div > div > div {
        color: white !important;
        font-weight: 600;
    }

    .stDateInput input {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: none !important;
        border-radius: 8px !important;
        color: #2c3e50 !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
    }

    /* Filter section styling */
    .filter-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }

    /* Headers and titles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 55px;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e3f2fd;
        transform: translateY(-1px);
    }

    /* Chart containers */
    .chart-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        border: 1px solid #e9ecef;
    }

    /* Filter labels */
    .filter-label {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        display: block;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    /* Date filter container */
    .date-filter-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }

    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
    }

    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }

    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }

    /* Sidebar text styling */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)


# Function to load data from Google Sheets
@st.cache_data(ttl=300)
def load_data_from_sheets(_credentials_dict):
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        credentials = Credentials.from_service_account_info(_credentials_dict, scopes=scopes)
        client = gspread.authorize(credentials)

        # Sales sheet IDs and names
        sales_sheets = [
            ("1Y0zoq6NODeLwOgftWwe4ooL687BTd24WRi9PO1GPmiA", "Master"),  # Jul-Sep 2025
            ("1QxhjtWwPsGrSUKZPD-tCNG9AnZ8wl2902vL1xAQsIaE", "Master"),  # Apr-Jun 2025
            ("1gSoF4Eox0C1UBk4aVy9SFi5uMAI2BWVdG3RKEcMljNI", "Master"),  # Oct-Dec 2024
            ("1nMvrCdTq3IJbuhoVcZzFsJzMGvZZG8U8q1x4Vn6Usvs", "Master"),  # Jan-Mar 2025
        ]

        # Load sales data with enhanced progress indication
        sales_dfs = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (sheet_id, sheet_name) in enumerate(sales_sheets):
            status_text.markdown(
                f'<div class="status-badge status-warning">Loading sales data {i + 1}/{len(sales_sheets)}...</div>',
                unsafe_allow_html=True)
            try:
                sheet = client.open_by_key(sheet_id).worksheet(sheet_name)
                data = sheet.get_all_records()
                df = pd.DataFrame(data)
                # Remove empty columns
                df = df.loc[:, (df != '').any(axis=0)]
                if not df.empty:
                    sales_dfs.append(df)
                progress_bar.progress((i + 1) / (len(sales_sheets) + 1))
            except Exception as e:
                st.warning(f"Could not load sales sheet {i + 1}: {str(e)}")

        # Load product data
        status_text.markdown('<div class="status-badge status-warning">Loading product data...</div>',
                             unsafe_allow_html=True)
        try:
            product_sheet = client.open_by_key("1TehvfsbUaSMWxe6-XNW8hJNkGykDCN-KWjFN-79TmKQ").worksheet("Sheet1")
            product_data = product_sheet.get_all_records()
            product_df = pd.DataFrame(product_data)
            progress_bar.progress(1.0)
            status_text.markdown('<div class="status-badge status-success">✅ Data loading complete!</div>',
                                 unsafe_allow_html=True)

            # Clear progress indicators
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

        except Exception as e:
            st.error(f"Could not load product data: {str(e)}")
            return [], pd.DataFrame()

        return sales_dfs, product_df

    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return [], pd.DataFrame()


def parse_dates_improved(sales_df):
    """
    Improved date parsing function that eliminates warnings and handles multiple formats efficiently
    """
    # Store original dates
    sales_df['Date_Original'] = sales_df['Date'].copy()

    # Define date formats in order of likelihood (most common first for performance)
    date_formats = [
        "%d-%b-%Y",  # 15-Jan-2024
        "%Y-%m-%d",  # 2024-01-15
        "%d/%m/%Y",  # 15/01/2024
        "%m/%d/%Y",  # 01/15/2024
        "%d-%m-%Y",  # 15-01-2024
        "%Y/%m/%d",  # 2024/01/15
        "%b %d, %Y",  # Jan 15, 2024
        "%B %d, %Y",  # January 15, 2024
        "%d %b %Y",  # 15 Jan 2024
        "%d %B %Y",  # 15 January 2024
    ]

    # Initialize Date column as NaT (Not a Time)
    sales_df['Date'] = pd.NaT

    # Track parsing success
    successfully_parsed = 0
    total_rows = len(sales_df)

    # Try each format systematically
    for fmt in date_formats:
        # Only process rows that haven't been successfully parsed yet
        mask = sales_df['Date'].isna()
        if not mask.any():
            break  # All dates parsed successfully

        try:
            # Convert the remaining unparsed dates
            parsed_dates = pd.to_datetime(
                sales_df.loc[mask, 'Date_Original'],
                format=fmt,
                errors='coerce'
            )

            # Update only the successfully parsed dates
            valid_dates_mask = parsed_dates.notna()
            if valid_dates_mask.any():
                sales_df.loc[mask, 'Date'] = parsed_dates.loc[valid_dates_mask]
                successfully_parsed += valid_dates_mask.sum()

        except (ValueError, TypeError):
            continue

    # Final attempt with automatic parsing for remaining dates (suppress warnings)
    mask = sales_df['Date'].isna()
    if mask.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                final_parsed = pd.to_datetime(
                    sales_df.loc[mask, 'Date_Original'],
                    errors='coerce',
                    infer_datetime_format=True  # Let pandas infer format
                )
                sales_df.loc[mask, 'Date'] = final_parsed
                successfully_parsed += final_parsed.notna().sum()
            except:
                pass  # Some dates might remain unparsed

    # Report parsing results
    failed_parsing = total_rows - successfully_parsed
    if failed_parsing > 0:
        st.warning(f"⚠️ Could not parse {failed_parsing} out of {total_rows} dates. These rows will be removed.")

        # Show problematic dates in an expander
        unparsed_dates = sales_df[sales_df['Date'].isna()]['Date_Original'].unique()
        if len(unparsed_dates) > 0:
            with st.expander("View Unparsed Dates", expanded=False):
                st.write("These date formats could not be parsed:")
                st.write(unparsed_dates[:10])  # Show first 10
                if len(unparsed_dates) > 10:
                    st.write(f"... and {len(unparsed_dates) - 10} more")

    return sales_df


# Enhanced data processing function
@st.cache_data
def process_data(sales_dfs, product_df):
    if not sales_dfs or product_df.empty:
        return pd.DataFrame()

    # Combine sales dataframes
    sales_df = pd.concat(sales_dfs, ignore_index=True)

    # Validate required columns
    required_sales_cols = ['Date', 'ID', 'City', 'Brand', 'Qty']
    required_product_cols = ['ID', 'Sale Price', 'Platform']

    missing_sales = [col for col in required_sales_cols if col not in sales_df.columns]
    missing_product = [col for col in required_product_cols if col not in product_df.columns]

    if missing_sales or missing_product:
        st.error(f"Missing columns - Sales: {missing_sales}, Product: {missing_product}")
        return pd.DataFrame()

    # Clean and convert data types
    merge_key = 'ID'

    # Convert ID to numeric
    sales_df[merge_key] = pd.to_numeric(sales_df[merge_key], errors='coerce')
    product_df[merge_key] = pd.to_numeric(product_df[merge_key], errors='coerce')

    # Convert quantity and prices
    sales_df['Qty'] = pd.to_numeric(sales_df['Qty'], errors='coerce').fillna(0)
    product_df['Sale Price'] = pd.to_numeric(product_df['Sale Price'], errors='coerce').fillna(0)

    if 'Cost Price' in product_df.columns:
        product_df['Cost Price'] = pd.to_numeric(product_df['Cost Price'], errors='coerce').fillna(0)

    # IMPROVED DATE PARSING
    sales_df = parse_dates_improved(sales_df)

    # Remove rows with invalid dates
    initial_rows = len(sales_df)
    sales_df = sales_df.dropna(subset=['Date'])

    if len(sales_df) == 0:
        st.error("No valid dates found in sales data")
        return pd.DataFrame()

    # Merge sales and product data
    merge_columns = [col for col in product_df.columns if col in [
        'ID', 'DesiDiya - SKU', 'Category Name', 'Sub-Category Name',
        'Sale Price', 'Platform', 'Cost Price', 'Product Name'
    ]]

    merged_df = pd.merge(sales_df, product_df[merge_columns], on=merge_key, how='left')

    # Handle missing product information
    merged_df['Platform'] = merged_df['Platform'].fillna('Unknown')
    merged_df['Category Name'] = merged_df['Category Name'].fillna('Unknown')
    merged_df['Sub-Category Name'] = merged_df['Sub-Category Name'].fillna('Unknown')
    merged_df['Sale Price'] = merged_df['Sale Price'].fillna(0)

    # Create Product Name from DesiDiya - SKU if available
    if 'DesiDiya - SKU' in merged_df.columns:
        merged_df['Product Name'] = merged_df['DesiDiya - SKU'].fillna('Unknown Product')
    elif 'Product Name' not in merged_df.columns:
        merged_df['Product Name'] = 'Unknown Product'
    else:
        merged_df['Product Name'] = merged_df['Product Name'].fillna('Unknown Product')

    # Calculate metrics
    merged_df['Total Sales'] = merged_df['Qty'] * merged_df['Sale Price']

    if 'Cost Price' in merged_df.columns:
        merged_df['Cost Price'] = merged_df['Cost Price'].fillna(0)
        merged_df['Profit'] = (merged_df['Sale Price'] - merged_df['Cost Price']) * merged_df['Qty']
        merged_df['Profit Margin %'] = np.where(
            merged_df['Sale Price'] > 0,
            (merged_df['Profit'] / merged_df['Total Sales']) * 100,
            0
        )

    # Clean up
    merged_df = merged_df.drop(columns=['Date_Original'], errors='ignore')

    return merged_df


def create_comprehensive_charts(df):
    """Create comprehensive visualizations"""

    # 1. Daily Sales Trend
    daily_data = df.groupby(df['Date'].dt.date).agg({
        'Total Sales': 'sum',
        'Qty': 'sum',
        'ID': 'count'  # Number of orders
    }).reset_index()
    daily_data.columns = ['Date', 'Sales', 'Quantity', 'Orders']

    # 2. Category Performance
    category_data = df.groupby('Category Name').agg({
        'Total Sales': 'sum',
        'Qty': 'sum',
        'Product Name': 'nunique'
    }).reset_index().sort_values('Total Sales', ascending=False)
    category_data.columns = ['Category', 'Sales', 'Quantity', 'Products']

    # 3. Platform Performance
    platform_data = df.groupby('Platform').agg({
        'Total Sales': 'sum',
        'Qty': 'sum',
        'Product Name': 'nunique'
    }).reset_index()
    platform_data.columns = ['Platform', 'Sales', 'Quantity', 'Products']

    # 4. City Performance
    city_data = df.groupby('City').agg({
        'Total Sales': 'sum',
        'Qty': 'sum'
    }).reset_index().sort_values('Total Sales', ascending=False)
    city_data.columns = ['City', 'Sales', 'Quantity']

    # 5. Brand Performance
    brand_data = df.groupby('Brand').agg({
        'Total Sales': 'sum',
        'Qty': 'sum',
        'Product Name': 'nunique'
    }).reset_index().sort_values('Total Sales', ascending=False)
    brand_data.columns = ['Brand', 'Sales', 'Quantity', 'Products']

    return daily_data, category_data, platform_data, city_data, brand_data


def display_advanced_kpis(df):
    """Display comprehensive KPIs with enhanced styling"""

    # Basic metrics
    total_sales = df['Total Sales'].sum()
    total_qty = df['Qty'].sum()
    total_orders = len(df)
    unique_products = df['Product Name'].nunique()
    unique_cities = df['City'].nunique()
    unique_platforms = df['Platform'].nunique()

    # Advanced metrics
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0
    avg_selling_price = total_sales / total_qty if total_qty > 0 else 0

    # Date range metrics
    date_range_days = (df['Date'].max() - df['Date'].min()).days + 1
    daily_avg_sales = total_sales / date_range_days if date_range_days > 0 else 0

    # First row of metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("💰 Total Sales", f"₹{total_sales:,.0f}")
    with col2:
        st.metric("📦 Total Quantity", f"{total_qty:,}")
    with col3:
        st.metric("🛒 Total Orders", f"{total_orders:,}")
    with col4:
        st.metric("🎯 Avg Order Value", f"₹{avg_order_value:,.0f}")
    with col5:
        st.metric("🏷️ Avg Selling Price", f"₹{avg_selling_price:.0f}")

    # Second row of metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🎨 Unique Products", f"{unique_products:,}")
    with col2:
        st.metric("🏙️ Cities", f"{unique_cities:,}")
    with col3:
        st.metric("🔗 Platforms", f"{unique_platforms:,}")
    with col4:
        st.metric("📅 Days Range", f"{date_range_days:,}")
    with col5:
        st.metric("📈 Daily Avg Sales", f"₹{daily_avg_sales:,.0f}")

    # Profit metrics if available
    if 'Profit' in df.columns and df['Profit'].sum() > 0:
        st.markdown("---")
        total_profit = df['Profit'].sum()
        profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
        avg_profit_per_order = total_profit / total_orders if total_orders > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💎 Total Profit", f"₹{total_profit:,.0f}")
        with col2:
            st.metric("📊 Profit Margin", f"{profit_margin:.1f}%")
        with col3:
            st.metric("💰 Avg Profit/Order", f"₹{avg_profit_per_order:,.0f}")


def create_charts_section(df):
    """Create comprehensive charts section with enhanced styling"""

    daily_data, category_data, platform_data, city_data, brand_data = create_comprehensive_charts(df)

    # Sales Trend Chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📈 Sales Trends Over Time")

    # Create subplot for sales trends
    fig_trends = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Daily Sales Revenue', 'Daily Quantity Sold', 'Daily Orders'),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )

    fig_trends.add_trace(
        go.Scatter(x=daily_data['Date'], y=daily_data['Sales'],
                   mode='lines+markers', name='Sales', line=dict(color='#1f77b4', width=3),
                   marker=dict(size=6)),
        row=1, col=1
    )

    fig_trends.add_trace(
        go.Scatter(x=daily_data['Date'], y=daily_data['Quantity'],
                   mode='lines+markers', name='Quantity', line=dict(color='#ff7f0e', width=3),
                   marker=dict(size=6)),
        row=2, col=1
    )

    fig_trends.add_trace(
        go.Scatter(x=daily_data['Date'], y=daily_data['Orders'],
                   mode='lines+markers', name='Orders', line=dict(color='#2ca02c', width=3),
                   marker=dict(size=6)),
        row=3, col=1
    )

    fig_trends.update_layout(
        height=800,
        showlegend=False,
        title_text="Sales Performance Trends",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig_trends.update_yaxes(title_text="Sales (₹)", row=1, col=1)
    fig_trends.update_yaxes(title_text="Quantity", row=2, col=1)
    fig_trends.update_yaxes(title_text="Orders", row=3, col=1)

    st.plotly_chart(fig_trends, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Category and Platform Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("📦 Top Categories by Sales")
        fig_cat_bar = px.bar(
            category_data.head(10),
            x='Sales', y='Category',
            orientation='h',
            title="Top 10 Categories by Revenue",
            color='Sales',
            color_continuous_scale='Blues'
        )
        fig_cat_bar.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cat_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🔗 Platform Distribution")
        fig_platform_pie = px.pie(
            platform_data,
            values='Sales',
            names='Platform',
            title="Sales Distribution by Platform",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_platform_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_platform_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_platform_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Geographic and Brand Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🏙️ Top Cities by Sales")
        fig_city = px.bar(
            city_data.head(15),
            x='City', y='Sales',
            title="Top 15 Cities by Revenue",
            color='Sales',
            color_continuous_scale='Greens'
        )
        fig_city.update_xaxes(tickangle=45)
        fig_city.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_city, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🏢 Brand Performance")
        fig_brand_scatter = px.scatter(
            brand_data,
            x='Quantity', y='Sales',
            size='Products',
            hover_name='Brand',
            title="Brand Performance Matrix",
            color='Sales',
            color_continuous_scale='Reds'
        )
        fig_brand_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_brand_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Monthly Trends
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("📅 Monthly Performance")

    df['Month'] = df['Date'].dt.to_period('M')
    monthly_data = df.groupby('Month').agg({
        'Total Sales': 'sum',
        'Qty': 'sum',
        'ID': 'count'
    }).reset_index()
    monthly_data['Month'] = monthly_data['Month'].astype(str)

    fig_monthly = px.line(
        monthly_data,
        x='Month', y=['Total Sales', 'Qty'],
        title="Monthly Sales and Quantity Trends",
        markers=True
    )
    fig_monthly.update_layout(
        yaxis_title="Value",
        xaxis_title="Month",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def create_enhanced_sidebar_filters(df):
    """Create enhanced sidebar with beautiful filters"""

    with st.sidebar:
        # Header with icon
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0; font-size: 2rem;">🎛️</h1>
            <h2 style="color: white; margin: 0.5rem 0 0 0; font-size: 1.5rem;">Filters</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Customize your data view</p>
        </div>
        """, unsafe_allow_html=True)

        # Date range filter with enhanced styling
        st.markdown("""
        <div class="date-filter-container">
            <div class="filter-label">📅 Date Range</div>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.85rem; margin: 0.5rem 0;">
                Select the time period for analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Date range
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()

        # Display date range info
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0;">
            <small style="color: rgba(255,255,255,0.9);">
                📊 Available data: <strong>{min_date}</strong> to <strong>{max_date}</strong><br>
                🗓️ Total days: <strong>{(max_date - min_date).days + 1}</strong>
            </small>
        </div>
        """, unsafe_allow_html=True)

        date_range = st.date_input(
            "Select Date Range",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date,
            help="Choose start and end dates for your analysis"
        )

        if len(date_range) != 2:
            st.error("⚠️ Please select both start and end dates")
            return None, None, None, None, None

        # Enhanced filter sections
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">🏙️ Cities</div>', unsafe_allow_html=True)
        cities = st.multiselect(
            "Select Cities",
            options=sorted(df['City'].unique()),
            default=df['City'].unique(),
            help="Filter by specific cities"
        )
        st.markdown(
            f'<small style="color: rgba(255,255,255,0.7);">Selected: {len(cities)} of {df["City"].nunique()} cities</small>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">🔗 Platforms</div>', unsafe_allow_html=True)
        platforms = st.multiselect(
            "Select Platforms",
            options=sorted(df['Platform'].unique()),
            default=df['Platform'].unique(),
            help="Filter by sales platforms"
        )
        st.markdown(
            f'<small style="color: rgba(255,255,255,0.7);">Selected: {len(platforms)} of {df["Platform"].nunique()} platforms</small>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">📦 Categories</div>', unsafe_allow_html=True)
        categories = st.multiselect(
            "Select Categories",
            options=sorted(df['Category Name'].unique()),
            default=df['Category Name'].unique(),
            help="Filter by product categories"
        )
        st.markdown(
            f'<small style="color: rgba(255,255,255,0.7);">Selected: {len(categories)} of {df["Category Name"].nunique()} categories</small>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Additional filters
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="filter-label">🏢 Brands</div>', unsafe_allow_html=True)
        brands = st.multiselect(
            "Select Brands",
            options=sorted(df['Brand'].unique()),
            default=df['Brand'].unique(),
            help="Filter by brands"
        )
        st.markdown(
            f'<small style="color: rgba(255,255,255,0.7);">Selected: {len(brands)} of {df["Brand"].nunique()} brands</small>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Quick filter buttons
        st.markdown("---")
        st.markdown('<div class="filter-label">⚡ Quick Filters</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📅 Last 30 Days", use_container_width=True):
                st.session_state.quick_filter = "30days"
        with col2:
            if st.button("📊 All Data", use_container_width=True):
                st.session_state.quick_filter = "all"

        # Filter summary
        st.markdown("---")
        total_records = len(df)
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px;">
            <h4 style="color: white; margin: 0 0 0.5rem 0;">📊 Filter Summary</h4>
            <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;">
                Total Records: <strong>{total_records:,}</strong><br>
                Date Range: <strong>{(date_range[1] - date_range[0]).days + 1}</strong> days
            </p>
        </div>
        """, unsafe_allow_html=True)

        return date_range, cities, platforms, categories, brands


# Main dashboard
def main():
    # Header with enhanced styling
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 20px 20px;">
        <h1 style="color: white; margin: 0; font-size: 3rem; text-shadow: 0 4px 8px rgba(0,0,0,0.3);">📊 Sales Analytics Dashboard</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.2rem;">Comprehensive Sales Data Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # Use Streamlit secrets for credentials
    try:
        credentials_dict = st.secrets["gcp_service_account"]

        # Load and process data
        with st.spinner("🔄 Loading data from Google Sheets..."):
            sales_dfs, product_df = load_data_from_sheets(credentials_dict)
            df = process_data(sales_dfs, product_df)

        if df.empty:
            st.error("❌ No valid data available")
            return

        # Enhanced data summary
        with st.expander("📊 Data Overview & Health Check", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📋 Total Records", f"{len(df):,}")
            with col2:
                matched_products = df['Product Name'].notna().sum()
                match_rate = (matched_products / len(df)) * 100
                st.metric("🔗 Products Matched", f"{matched_products:,}")
                st.caption(f"Match rate: {match_rate:.1f}%")
            with col3:
                date_range = (df['Date'].max() - df['Date'].min()).days
                st.metric("📅 Date Range", f"{date_range} days")
                st.caption(f"From {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
            with col4:
                platforms = df['Platform'].nunique()
                st.metric("🔗 Platforms", platforms)
                platform_list = ", ".join(df['Platform'].unique()[:3])
                st.caption(f"Including: {platform_list}...")

        # Create enhanced sidebar filters
        filter_results = create_enhanced_sidebar_filters(df)
        if filter_results[0] is None:
            return

        date_range, cities, platforms, categories, brands = filter_results

        # Handle quick filters
        if hasattr(st.session_state, 'quick_filter'):
            if st.session_state.quick_filter == "30days":
                end_date = df['Date'].max().date()
                start_date = end_date - timedelta(days=30)
                date_range = [start_date, end_date]
                st.session_state.quick_filter = None  # Reset
            elif st.session_state.quick_filter == "all":
                date_range = [df['Date'].min().date(), df['Date'].max().date()]
                st.session_state.quick_filter = None  # Reset

        # Apply filters
        filtered_df = df[
            (df['City'].isin(cities)) &
            (df['Platform'].isin(platforms)) &
            (df['Category Name'].isin(categories)) &
            (df['Brand'].isin(brands)) &
            (df['Date'].dt.date >= date_range[0]) &
            (df['Date'].dt.date <= date_range[1])
            ]

        if filtered_df.empty:
            st.warning("⚠️ No data matches the selected filters. Please adjust your filter criteria.")
            return

        # Filter impact summary
        filter_impact = len(filtered_df) / len(df) * 100
        records_filtered = len(df) - len(filtered_df)

        if records_filtered > 0:
            st.info(
                f"🔍 **Filter Applied**: Showing {len(filtered_df):,} of {len(df):,} records ({filter_impact:.1f}%) | {records_filtered:,} records filtered out")

        # Display KPIs
        st.header("🎯 Key Performance Indicators")
        display_advanced_kpis(filtered_df)
        st.markdown("---")

        # Charts section
        st.header("📊 Analytics & Visualizations")
        create_charts_section(filtered_df)
        st.markdown("---")

        # Detailed tables in tabs
        st.header("📋 Detailed Analysis Tables")

        tab1, tab2, tab3, tab4 = st.tabs([
            "🛍️ Product Analysis",
            "🔗 Platform Analysis",
            "🏙️ Geographic Analysis",
            "📅 Time Analysis"
        ])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏆 Top Products by Sales")
                product_summary = filtered_df.groupby('Product Name').agg({
                    'Total Sales': 'sum',
                    'Qty': 'sum',
                    'ID': 'count'
                }).reset_index().sort_values('Total Sales', ascending=False)
                product_summary.columns = ['Product Name', 'Total Sales (₹)', 'Quantity Sold', 'Total Orders']
                product_summary['Total Sales (₹)'] = product_summary['Total Sales (₹)'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(product_summary.head(20), use_container_width=True, hide_index=True)

            with col2:
                st.subheader("📦 Category Performance")
                category_summary = filtered_df.groupby('Category Name').agg({
                    'Total Sales': 'sum',
                    'Qty': 'sum',
                    'Product Name': 'nunique'
                }).reset_index().sort_values('Total Sales', ascending=False)
                category_summary.columns = ['Category', 'Total Sales (₹)', 'Quantity', 'Unique Products']
                category_summary['Total Sales (₹)'] = category_summary['Total Sales (₹)'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(category_summary, use_container_width=True, hide_index=True)

        with tab2:
            st.subheader("🔗 Platform Performance Summary")
            platform_summary = filtered_df.groupby('Platform').agg({
                'Total Sales': 'sum',
                'Qty': 'sum',
                'Product Name': 'nunique',
                'City': 'nunique'
            }).reset_index()
            platform_summary.columns = ['Platform', 'Total Sales (₹)', 'Quantity', 'Products', 'Cities Covered']
            platform_summary['Total Sales (₹)'] = platform_summary['Total Sales (₹)'].apply(lambda x: f"₹{x:,.0f}")
            st.dataframe(platform_summary, use_container_width=True, hide_index=True)

            # Cross-tab: Product vs Platform
            st.subheader("📊 Product vs Platform Matrix")
            pivot_table = filtered_df.pivot_table(
                values='Total Sales',
                index='Product Name',
                columns='Platform',
                aggfunc='sum',
                fill_value=0
            ).round(0)
            # Format as currency
            pivot_table = pivot_table.applymap(lambda x: f"₹{x:,.0f}" if x > 0 else "₹0")
            st.dataframe(pivot_table, use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🏙️ City Performance")
                city_summary = filtered_df.groupby('City').agg({
                    'Total Sales': 'sum',
                    'Qty': 'sum',
                    'Product Name': 'nunique'
                }).reset_index().sort_values('Total Sales', ascending=False)
                city_summary.columns = ['City', 'Total Sales (₹)', 'Quantity', 'Unique Products']
                city_summary['Total Sales (₹)'] = city_summary['Total Sales (₹)'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(city_summary, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("🏢 Brand Performance")
                brand_summary = filtered_df.groupby('Brand').agg({
                    'Total Sales': 'sum',
                    'Qty': 'sum',
                    'City': 'nunique'
                }).reset_index().sort_values('Total Sales', ascending=False)
                brand_summary.columns = ['Brand', 'Total Sales (₹)', 'Quantity', 'Cities Covered']
                brand_summary['Total Sales (₹)'] = brand_summary['Total Sales (₹)'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(brand_summary, use_container_width=True, hide_index=True)

        with tab4:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📅 Monthly Summary")
                filtered_df['Month'] = filtered_df['Date'].dt.strftime('%Y-%m')
                monthly_summary = filtered_df.groupby('Month').agg({
                    'Total Sales': 'sum',
                    'Qty': 'sum',
                    'ID': 'count'
                }).reset_index()
                monthly_summary.columns = ['Month', 'Total Sales (₹)', 'Quantity', 'Orders']
                monthly_summary['Total Sales (₹)'] = monthly_summary['Total Sales (₹)'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(monthly_summary, use_container_width=True, hide_index=True)

            with col2:
                st.subheader("📊 Weekly Summary")
                filtered_df['Week'] = filtered_df['Date'].dt.strftime('%Y-W%U')
                weekly_summary = filtered_df.groupby('Week').agg({
                    'Total Sales': 'sum',
                    'Qty': 'sum',
                    'ID': 'count'
                }).reset_index().tail(10)
                weekly_summary.columns = ['Week', 'Total Sales (₹)', 'Quantity', 'Orders']
                weekly_summary['Total Sales (₹)'] = weekly_summary['Total Sales (₹)'].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(weekly_summary, use_container_width=True, hide_index=True)

        # Enhanced export section
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #28a745 0%, #20c997 100%); border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: white; margin: 0 0 0.5rem 0;">💾 Export Your Data</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0;">Download filtered data for further analysis</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Create summary CSV
            summary_data = {
                'Metric': ['Total Sales', 'Total Quantity', 'Total Orders', 'Unique Products', 'Date Range'],
                'Value': [
                    f"₹{filtered_df['Total Sales'].sum():,.0f}",
                    f"{filtered_df['Qty'].sum():,}",
                    f"{len(filtered_df):,}",
                    f"{filtered_df['Product Name'].nunique():,}",
                    f"{(date_range[1] - date_range[0]).days + 1} days"
                ]
            }
            summary_csv = pd.DataFrame(summary_data).to_csv(index=False)
            st.download_button(
                label="📊 Download Summary",
                data=summary_csv,
                file_name=f"sales_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col3:
            if st.button("👁️ Preview Data", use_container_width=True):
                st.session_state.show_preview = not getattr(st.session_state, 'show_preview', False)

        # Show data preview if requested
        if getattr(st.session_state, 'show_preview', False):
            st.markdown("---")
            st.subheader("👁️ Data Preview")
            st.markdown(f"Showing first 100 rows of {len(filtered_df):,} filtered records")

            # Format the preview data
            preview_df = filtered_df.head(100).copy()
            if 'Total Sales' in preview_df.columns:
                preview_df['Total Sales'] = preview_df['Total Sales'].apply(lambda x: f"₹{x:,.0f}")
            if 'Sale Price' in preview_df.columns:
                preview_df['Sale Price'] = preview_df['Sale Price'].apply(lambda x: f"₹{x:,.0f}")

            st.dataframe(preview_df, use_container_width=True, hide_index=True)

        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1rem; color: #6c757d; font-size: 0.9rem;">
            <p>📊 Sales Analytics Dashboard | Last updated: {}</p>
            <p>💡 Use the sidebar filters to customize your analysis | 🔄 Data refreshes every 5 minutes</p>
        </div>
        """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Check your credentials in Streamlit secrets and data sources")

        # Error details in expander
        with st.expander("🔧 Troubleshooting Details"):
            st.code(str(e))
            st.markdown("""
            **Common solutions:**
            - Verify Google Sheets credentials in Streamlit secrets
            - Check internet connectivity
            - Ensure sheets are accessible and shared properly
            - Verify sheet IDs in the code
            """)


if __name__ == "__main__":
    main()