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

# Custom CSS for better styling
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Function to load data from Google Sheets
@st.cache_data(ttl=300)
def load_data_from_sheets(credentials_json):
    try:
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        credentials = Credentials.from_service_account_info(credentials_json, scopes=scopes)
        client = gspread.authorize(credentials)

        # Sales sheet IDs and names
        sales_sheets = [
            ("1Y0zoq6NODeLwOgftWwe4ooL687BTd24WRi9PO1GPmiA", "Master"),  # Jul-Sep 2025
            ("1QxhjtWwPsGrSUKZPD-tCNG9AnZ8wl2902vL1xAQsIaE", "Master"),  # Apr-Jun 2025
            ("1gSoF4Eox0C1UBk4aVy9SFi5uMAI2BWVdG3RKEcMljNI", "Master"),  # Oct-Dec 2024
            ("1nMvrCdTq3IJbuhoVcZzFsJzMGvZZG8U8q1x4Vn6Usvs", "Master"),  # Jan-Mar 2025
        ]

        # Load sales data with progress indication
        sales_dfs = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (sheet_id, sheet_name) in enumerate(sales_sheets):
            status_text.text(f'Loading sales data {i + 1}/{len(sales_sheets)}...')
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
        status_text.text('Loading product data...')
        try:
            product_sheet = client.open_by_key("1TehvfsbUaSMWxe6-XNW8hJNkGykDCN-KWjFN-79TmKQ").worksheet("Sheet1")
            product_data = product_sheet.get_all_records()
            product_df = pd.DataFrame(product_data)
            progress_bar.progress(1.0)
            status_text.text('Data loading complete!')

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
    """Display comprehensive KPIs"""

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
    """Create comprehensive charts section"""

    daily_data, category_data, platform_data, city_data, brand_data = create_comprehensive_charts(df)

    # Sales Trend Chart
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
                   mode='lines+markers', name='Sales', line=dict(color='#1f77b4', width=2)),
        row=1, col=1
    )

    fig_trends.add_trace(
        go.Scatter(x=daily_data['Date'], y=daily_data['Quantity'],
                   mode='lines+markers', name='Quantity', line=dict(color='#ff7f0e', width=2)),
        row=2, col=1
    )

    fig_trends.add_trace(
        go.Scatter(x=daily_data['Date'], y=daily_data['Orders'],
                   mode='lines+markers', name='Orders', line=dict(color='#2ca02c', width=2)),
        row=3, col=1
    )

    fig_trends.update_layout(height=800, showlegend=False, title_text="Sales Performance Trends")
    fig_trends.update_yaxes(title_text="Sales (₹)", row=1, col=1)
    fig_trends.update_yaxes(title_text="Quantity", row=2, col=1)
    fig_trends.update_yaxes(title_text="Orders", row=3, col=1)

    st.plotly_chart(fig_trends, use_container_width=True)

    # Category Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Top Categories by Sales")
        fig_cat_bar = px.bar(
            category_data.head(10),
            x='Sales', y='Category',
            orientation='h',
            title="Top 10 Categories by Revenue",
            color='Sales',
            color_continuous_scale='Blues'
        )
        fig_cat_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_cat_bar, use_container_width=True)

    with col2:
        st.subheader("🔗 Platform Distribution")
        fig_platform_pie = px.pie(
            platform_data,
            values='Sales',
            names='Platform',
            title="Sales Distribution by Platform"
        )
        fig_platform_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_platform_pie, use_container_width=True)

    # Geographic Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏙️ Top Cities by Sales")
        fig_city = px.bar(
            city_data.head(15),
            x='City', y='Sales',
            title="Top 15 Cities by Revenue",
            color='Sales',
            color_continuous_scale='Greens'
        )
        fig_city.update_xaxes(tickangle=45)
        st.plotly_chart(fig_city, use_container_width=True)

    with col2:
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
        st.plotly_chart(fig_brand_scatter, use_container_width=True)

    # Monthly Trends
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
    fig_monthly.update_layout(yaxis_title="Value", xaxis_title="Month")
    st.plotly_chart(fig_monthly, use_container_width=True)


# Main dashboard
def main():
    st.title("📊 Sales Analytics Dashboard")
    st.markdown("### Comprehensive Sales Data Analysis Platform")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Configuration")

        with st.expander("📋 Google Sheets Credentials", expanded=True):
            credentials_json = st.text_area(
                "Paste Google Service Account JSON:",
                height=150,
                placeholder="Paste your service account JSON here...",
                help="Get credentials from Google Cloud Console"
            )

        if not credentials_json:
            st.info("👆 Please provide credentials to continue")
            st.markdown("""
            ### 📋 Setup Instructions:
            1. Create Google Cloud Project
            2. Enable Google Sheets API
            3. Create Service Account
            4. Download JSON credentials
            5. Paste content above
            """)
            return

    try:
        credentials_dict = json.loads(credentials_json)

        # Load and process data
        with st.spinner("🔄 Loading data from Google Sheets..."):
            sales_dfs, product_df = load_data_from_sheets(credentials_dict)
            df = process_data(sales_dfs, product_df)

        if df.empty:
            st.error("❌ No valid data available")
            return

        # Data summary
        with st.expander("📊 Data Summary", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                matched_products = df['Product Name'].notna().sum()
                st.metric("Products Matched", matched_products)
            with col3:
                date_range = (df['Date'].max() - df['Date'].min()).days
                st.metric("Date Range (Days)", date_range)
            with col4:
                platforms = df['Platform'].nunique()
                st.metric("Platforms", platforms)

        # Sidebar filters
        with st.sidebar:
            st.header("🎛️ Filters")

            # Date range
            min_date = df['Date'].min().date()
            max_date = df['Date'].max().date()
            date_range = st.date_input(
                "📅 Date Range",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )

            if len(date_range) != 2:
                st.warning("Select both start and end dates")
                return

            # Other filters
            cities = st.multiselect(
                "🏙️ Cities",
                options=sorted(df['City'].unique()),
                default=df['City'].unique()
            )

            platforms = st.multiselect(
                "🔗 Platforms",
                options=sorted(df['Platform'].unique()),
                default=df['Platform'].unique()
            )

            categories = st.multiselect(
                "📦 Categories",
                options=sorted(df['Category Name'].unique()),
                default=df['Category Name'].unique()
            )

        # Apply filters
        filtered_df = df[
            (df['City'].isin(cities)) &
            (df['Platform'].isin(platforms)) &
            (df['Category Name'].isin(categories)) &
            (df['Date'].dt.date >= date_range[0]) &
            (df['Date'].dt.date <= date_range[1])
            ]

        if filtered_df.empty:
            st.warning("⚠️ No data matches the selected filters")
            return

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
            st.subheader("Top Products by Sales")
            product_summary = filtered_df.groupby('Product Name').agg({
                'Total Sales': 'sum',
                'Qty': 'sum',
                'ID': 'count'
            }).reset_index().sort_values('Total Sales', ascending=False)
            product_summary.columns = ['Product Name', 'Total Sales', 'Quantity', 'Orders']
            st.dataframe(product_summary.head(20), use_container_width=True)

            st.subheader("Category Performance")
            category_summary = filtered_df.groupby('Category Name').agg({
                'Total Sales': 'sum',
                'Qty': 'sum',
                'Product Name': 'nunique'
            }).reset_index().sort_values('Total Sales', ascending=False)
            category_summary.columns = ['Category', 'Total Sales', 'Quantity', 'Unique Products']
            st.dataframe(category_summary, use_container_width=True)

        with tab2:
            st.subheader("Platform Performance Summary")
            platform_summary = filtered_df.groupby('Platform').agg({
                'Total Sales': 'sum',
                'Qty': 'sum',
                'Product Name': 'nunique',
                'City': 'nunique'
            }).reset_index()
            platform_summary.columns = ['Platform', 'Total Sales', 'Quantity', 'Products', 'Cities']
            st.dataframe(platform_summary, use_container_width=True)

            # Cross-tab: Product vs Platform
            st.subheader("Product vs Platform Matrix")
            pivot_table = filtered_df.pivot_table(
                values='Total Sales',
                index='Product Name',
                columns='Platform',
                aggfunc='sum',
                fill_value=0
            ).round(0)
            st.dataframe(pivot_table, use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("City Performance")
                city_summary = filtered_df.groupby('City').agg({
                    'Total Sales': 'sum',
                    'Qty': 'sum',
                    'Product Name': 'nunique'
                }).reset_index().sort_values('Total Sales', ascending=False)
                city_summary.columns = ['City', 'Total Sales', 'Quantity', 'Unique Products']
                st.dataframe(city_summary, use_container_width=True)

            with col2:
                st.subheader("Brand Performance")
                brand_summary = filtered_df.groupby('Brand').agg({
                    'Total Sales': 'sum',
                    'Qty': 'sum',
                    'City': 'nunique'
                }).reset_index().sort_values('Total Sales', ascending=False)
                brand_summary.columns = ['Brand', 'Total Sales', 'Quantity', 'Cities']
                st.dataframe(brand_summary, use_container_width=True)

        with tab4:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Monthly Summary")
                filtered_df['Month'] = filtered_df['Date'].dt.strftime('%Y-%m')
                monthly_summary = filtered_df.groupby('Month').agg({
                    'Total Sales': 'sum',
                    'Qty': 'sum',
                    'ID': 'count'
                }).reset_index()
                monthly_summary.columns = ['Month', 'Total Sales', 'Quantity', 'Orders']
                st.dataframe(monthly_summary, use_container_width=True)

            with col2:
                st.subheader("Weekly Summary")
                filtered_df['Week'] = filtered_df['Date'].dt.strftime('%Y-W%U')
                weekly_summary = filtered_df.groupby('Week').agg({
                    'Total Sales': 'sum',
                    'Qty': 'sum',
                    'ID': 'count'
                }).reset_index().tail(10)
                weekly_summary.columns = ['Week', 'Total Sales', 'Quantity', 'Orders']
                st.dataframe(weekly_summary, use_container_width=True)

        # Export section
        st.markdown("---")
        st.header("💾 Export Data")

        col1, col2 = st.columns(2)

        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f"sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        with col2:
            if st.checkbox("👁️ Show Raw Data Sample"):
                st.dataframe(filtered_df.head(100), use_container_width=True)

    except json.JSONDecodeError:
        st.error("❌ Invalid JSON format in credentials")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("Check your credentials and data sources")


if __name__ == "__main__":
    main()