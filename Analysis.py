# ===================== 1. IMPORT LIBRARIES =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

pd.options.display.float_format = '${:,.2f}'.format

# ===================== 2. LOAD & CLEAN DATA =====================
orders = pd.read_csv("Cleaned.CustomerOrderRows.csv")
customers = pd.read_csv("Cleaned.Customers.csv")

orders['OrderDate'] = pd.to_datetime(orders['OrderDate'], errors='coerce')
orders['Revenue'] = pd.to_numeric(orders['Revenue'], errors='coerce')
orders['BusinessContactId'] = pd.to_numeric(orders['BusinessContactId'], errors='coerce')
customers['BusinessContactId'] = pd.to_numeric(customers['BusinessContactId'], errors='coerce')

# Global snapshot date
snapshot_date = orders['OrderDate'].max() + pd.Timedelta(days=1)

# ===================== 3. MERGE DATA =====================
merged_df = pd.merge(orders, customers, on='BusinessContactId', how='left')

# ===================== 4. REVENUE STATUS ANALYSIS =====================
status_summary = merged_df.groupby('Status')['Revenue'].agg(
    total='sum',
    positive=lambda x: x[x > 0].sum(),
    negative=lambda x: x[x < 0].sum(),
    count='count'
).reset_index()

# ===================== 5. TOP CUSTOMERS BY QUANTITY =====================
top_customers_position = merged_df.groupby(['BusinessContactId', 'Name']).agg(
    total_orders=('OrderNumber', 'nunique'),
    total_quantity=('Position', 'sum'),
    total_revenue=('Revenue', 'sum')
).sort_values(by='total_quantity', ascending=False).head(50).reset_index()

# ===================== 6. STATUS, DELIVERED, QC COUNTS =====================
status_counts = merged_df['Status'].value_counts().reset_index()
status_counts.columns = ['Status', 'Count']

delivered_counts = merged_df['Delivered'].value_counts().reset_index()
delivered_counts.columns = ['Delivered', 'Count']

qcuse_counts = merged_df['QCUse'].value_counts().reset_index()
qcuse_counts.columns = ['QCUse', 'Count']

# ===================== 7. TOTAL REVENUE SUMMARY =====================
positive_revenue = merged_df[merged_df['Revenue'] > 0]['Revenue'].sum()
negative_revenue = merged_df[merged_df['Revenue'] < 0]['Revenue'].sum()
net_revenue = merged_df['Revenue'].sum()

revenue_summary = pd.DataFrame({
    'Metric': ['Positive Revenue', 'Negative Revenue', 'Net Revenue'],
    'Amount': [positive_revenue, negative_revenue, net_revenue]
})

# ===================== 8. RFM SEGMENTATION =====================
rfm = merged_df.groupby(['BusinessContactId', 'Name']).agg({
    'OrderDate': lambda x: (snapshot_date - x.max()).days,
    'OrderNumber': 'nunique',
    'Revenue': 'sum'
}).reset_index()

rfm.columns = ['BusinessContactId', 'Name', 'Recency', 'Frequency', 'Monetary']

rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4, 3, 2, 1]).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1, 2, 3, 4]).astype(int)

rfm['Segment'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# RFM Segment labeling
def label_segment(row):
    if row['Segment'] == '444': return 'Champions'
    elif row['R_Score'] >= 3 and row['F_Score'] >= 3: return 'Loyal Customers'
    elif row['R_Score'] == 4: return 'New Customers'
    elif row['R_Score'] <= 2 and row['F_Score'] <= 2: return 'At Risk'
    else: return 'Others'

rfm['Segment_Label'] = rfm.apply(label_segment, axis=1)

# Fill all customers in RFM base
rfm_base = customers[['BusinessContactId', 'Name']]
rfm_full = rfm_base.merge(rfm, on='BusinessContactId', how='left')

rfm_full[['Recency', 'Frequency', 'Monetary']] = rfm_full[['Recency', 'Frequency', 'Monetary']].fillna(0)
rfm_full[['R_Score', 'F_Score', 'M_Score']] = rfm_full[['R_Score', 'F_Score', 'M_Score']].fillna(0).astype(int)
rfm_full['Segment_Label'] = rfm_full['Segment_Label'].fillna('Inactive')
rfm_full['Engagement_Status'] = rfm_full['Segment_Label'].apply(lambda x: 'Engaged' if x != 'Inactive' else 'Inactive')

# ===================== 9. ENGAGEMENT SUMMARY =====================
engagement_summary = rfm_full.groupby('Engagement_Status').agg(
    Customer_Count=('BusinessContactId', 'count'),
    Total_Revenue=('Monetary', 'sum')
).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(engagement_summary['Engagement_Status'], engagement_summary['Customer_Count'], color=['steelblue', 'lightgray'])

for bar, (cust_count, revenue) in zip(bars, zip(engagement_summary['Customer_Count'], engagement_summary['Total_Revenue'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30, f"{cust_count:,} customers", ha='center', fontsize=10)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, f"${revenue:,.0f}", ha='center', fontsize=10, color='white')

ax.set_ylabel('Customer Count')
ax.set_title('Engaged vs Inactive Customers and Revenue')
plt.tight_layout()
plt.show()

# ===================== 10. COHORT RETENTION ANALYSIS =====================
orders_rfm = orders.merge(rfm_full[['BusinessContactId', 'Segment_Label']], on='BusinessContactId', how='left')
orders_rfm['OrderMonth'] = orders_rfm['OrderDate'].dt.to_period('M')
first_order = orders_rfm.groupby('BusinessContactId')['OrderMonth'].min()
orders_rfm['CohortMonth'] = orders_rfm['BusinessContactId'].map(first_order)

orders_rfm['CohortIndex'] = (
    (orders_rfm['OrderMonth'].dt.year - orders_rfm['CohortMonth'].dt.year) * 12 +
    (orders_rfm['OrderMonth'].dt.month - orders_rfm['CohortMonth'].dt.month) + 1
)

orders_rfm['CohortBucket'] = pd.cut(
    orders_rfm['CohortIndex'],
    bins=[1, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 999],
    labels=['0–18', '19–36', '37–54', '55–72', '73–90', '91–108', '109–126', '127–144', '145–162', '163–180', '181–198', '199+'],
    include_lowest=True
)

segment_cohort = orders_rfm.groupby(['Segment_Label', 'CohortBucket'])['BusinessContactId'].nunique().reset_index()
segment_pivot = segment_cohort.pivot(index='Segment_Label', columns='CohortBucket', values='BusinessContactId').fillna(0)
segment_retention = segment_pivot.div(segment_pivot.iloc[:, 0], axis=0).round(3)

plt.figure(figsize=(18, 6))
ax = sns.heatmap(segment_retention, cmap="YlGnBu", annot=False, cbar_kws={'label': 'Avg Retention Rate'})
for y in range(segment_retention.shape[0]):
    for x in range(segment_retention.shape[1]):
        val = segment_retention.iloc[y, x]
        ax.text(x + 0.5, y + 0.5, f"{val:.1%}", ha='center', va='center', color='white' if val > 0.3 else 'black')
plt.title("RFM Segment Retention (%) Across Time Buckets")
plt.ylabel("RFM Segment")
plt.xlabel("Months Since First Order (Bucketed)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ===================== 11. CHURN ANALYSIS =====================
last_order_df = orders.groupby('BusinessContactId')['OrderDate'].max().reset_index()
last_order_df.columns = ['BusinessContactId', 'LastOrderDate']
last_order_df['DaysSinceLastOrder'] = (snapshot_date - last_order_df['LastOrderDate']).dt.days

customer_days_df = customers[['BusinessContactId', 'Name']].merge(last_order_df, on='BusinessContactId', how='left')
customer_days_df['DaysSinceLastOrder'] = customer_days_df['DaysSinceLastOrder'].fillna(-1).astype(int)

def refined_churn_category(days):
    if days == -1: return 'Never Ordered'
    elif days <= 180: return 'Active'
    elif days <= 730: return 'Dormant'
    else: return 'Churned'

customer_days_df['RefinedChurnCategory'] = customer_days_df['DaysSinceLastOrder'].apply(refined_churn_category)

# ===================== 12. REVENUE BY CHURN CATEGORY =====================
orders_with_churn = orders.merge(customer_days_df[['BusinessContactId', 'RefinedChurnCategory']], on='BusinessContactId', how='left')
revenue_by_churn = orders_with_churn.groupby('RefinedChurnCategory')['Revenue'].sum().reset_index()
revenue_by_churn.columns = ['Customer Category', 'Total Revenue']

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=revenue_by_churn, x='Customer Category', y='Total Revenue', palette='Set3')
for bar in ax.patches:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 10000, f"${height:,.0f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.title('Total Revenue by Churn Category')
plt.xlabel('Customer Category')
plt.ylabel('Total Revenue ($)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ===================== 13. CHURN PREDICTION MODEL (LOGISTIC REGRESSION) =====================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Label churn (1 if 'Churned', 0 otherwise)
churn_df = customer_days_df.copy()
churn_df['Churned'] = churn_df['RefinedChurnCategory'].apply(lambda x: 1 if x == 'Churned' else 0)

# Aggregate features from order data
behavior_features = orders.groupby('BusinessContactId').agg(
    total_orders=('OrderNumber', 'nunique'),
    total_revenue=('Revenue', 'sum'),
    avg_order_value=('Revenue', 'mean'),
    total_discount=('Discount', 'sum'),
    avg_fulfillment_rate=('FulfillmentRate', 'mean'),
    qc_use_ratio=('QCUse', lambda x: x.sum() / len(x)),
    delivered_ratio=('Delivered', lambda x: x.sum() / len(x)),
    unique_parts=('PartId', pd.Series.nunique)
).reset_index()

# Merge features with churn labels
churn_df = churn_df.merge(behavior_features, on='BusinessContactId', how='left')
churn_df.fillna(0, inplace=True)

# Define feature set and target
feature_columns = [
    'total_orders', 'total_revenue', 'avg_order_value', 'total_discount',
    'avg_fulfillment_rate', 'qc_use_ratio', 'delivered_ratio', 'unique_parts'
]
X = churn_df[feature_columns]
y = churn_df['Churned']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Train logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Predict and evaluate
y_pred = logreg.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(class_report).transpose()

# Temporarily reset float formatting
pd.options.display.float_format = None

# Display classification report
print("📋 Logistic Regression Classification Report")
print(report_df)

# ===================== 15. FEATURE IMPORTANCE FROM LOGISTIC REGRESSION =====================
import matplotlib.pyplot as plt

# Retrieve feature names and coefficients
coefficients = logreg.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': coefficients,
    'AbsCoefficient': np.abs(coefficients)
}).sort_values(by='AbsCoefficient', ascending=False)

# Plotting feature importance
plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='skyblue')
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Logistic Regression Coefficients (Feature Impact on Churn)")
plt.xlabel("Coefficient Value (Positive = More Likely to Churn)")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop Features Driving Churn (with Coefficients):")
print(feature_importance[['Feature', 'Coefficient']])



# ===================== 16. RANDOM FOREST CLASSIFICATION MODEL =====================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
rf_pred = rf_model.predict(X_test)
rf_report = classification_report(y_test, rf_pred, output_dict=True)
rf_report_df = pd.DataFrame(rf_report).transpose()

# Display classification report
print("\n\ud83c\udf33 Random Forest Classification Report")
print(rf_report_df)

# ===================== 17. FEATURE IMPORTANCE FROM RANDOM FOREST =====================
rf_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(rf_importance['Feature'], rf_importance['Importance'], color='forestgreen')
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

########################################################################################

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Prepare customer-level metrics
customer_metrics = orders.groupby('BusinessContactId').agg(
    total_revenue=('Revenue', 'sum'),
    total_orders=('OrderNumber', 'nunique')
).reset_index()

# Step 2: Create churn status if not available
snapshot_date = orders['OrderDate'].max() + pd.Timedelta(days=1)
last_order_df = orders.groupby('BusinessContactId')['OrderDate'].max().reset_index()
last_order_df['DaysSinceLastOrder'] = (snapshot_date - last_order_df['OrderDate']).dt.days

def churn_category(days):
    if days <= 180:
        return 'Active'
    elif days <= 730:
        return 'Dormant'
    else:
        return 'Churned'

last_order_df['RefinedChurnCategory'] = last_order_df['DaysSinceLastOrder'].apply(churn_category)

# Step 3: Merge all together
prioritization_df = customer_metrics.merge(last_order_df[['BusinessContactId', 'RefinedChurnCategory']], on='BusinessContactId', how='left')
prioritization_df = prioritization_df.merge(customers[['BusinessContactId', 'Name']], on='BusinessContactId', how='left')

# Step 4: Assign clean priority labels
def assign_priority(row, threshold):
    if row['total_revenue'] >= threshold and row['RefinedChurnCategory'] == 'Churned':
        return 'High Value - At Risk'
    elif row['total_revenue'] >= threshold:
        return 'High Value - Retained'
    elif row['RefinedChurnCategory'] == 'Churned':
        return 'Low Value - Churned'
    else:
        return 'Regular'

revenue_threshold = customer_metrics['total_revenue'].quantile(0.9)
prioritization_df['Customer_Priority'] = prioritization_df.apply(assign_priority, axis=1, threshold=revenue_threshold)

# Step 5: Count customers in each segment
priority_counts = prioritization_df['Customer_Priority'].value_counts().reset_index()
priority_counts.columns = ['Customer_Priority', 'Customer_Count']

# ===================== Bar Chart =====================
plt.figure(figsize=(10, 5))
bars = plt.bar(priority_counts['Customer_Priority'], priority_counts['Customer_Count'], color='steelblue', edgecolor='black')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 5, f"{height:,}", ha='center', fontsize=10)

plt.title("Customer Prioritization Segments (Count)")
plt.xlabel("Priority Segment")
plt.ylabel("Number of Customers")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ===================== Pie Chart =====================
plt.figure(figsize=(8, 6))
plt.pie(priority_counts['Customer_Count'],
        labels=priority_counts['Customer_Priority'],
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops=dict(edgecolor='black'))

plt.title("Customer Priority Segment Share")
plt.tight_layout()
plt.show()



