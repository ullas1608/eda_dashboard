"""
Interactive Exploratory Data Analysis (EDA) Classroom Dashboard
==============================================================
A single-file educational web application built with Python, Flask, Pandas,
NumPy, Matplotlib, Seaborn, and vanilla HTML/CSS/JavaScript.

Dataset: Students Performance in Exams (Kaggle)
Author: Data Science Instructor
File: eda_dashboard.py
"""

import os
import io
import base64
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend for web server
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify, request, render_template_string

# ==============================================================================
# CONFIGURATION & GLOBAL SETTINGS
# ==============================================================================
CSV_FILE_PATH = "StudentsPerformance.csv"

# Global theme styling for teacher classroom aesthetic
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8


# ==============================================================================
# HELPER UTILITY: IN-MEMORY CHART ENCODING
# ==============================================================================
def fig_to_base64(fig):
    """
    Saves a Matplotlib figure to an in-memory BytesIO buffer,
    encodes it as a base64 string for direct browser rendering,
    and closes the figure to prevent server memory leaks.
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


# ==============================================================================
# 1. LOAD DATASET
# ==============================================================================
def load_dataset(filepath=CSV_FILE_PATH):
    """
    Loads the student performance dataset from a configurable CSV path.
    Checks the local working directory and script directory.
    Returns a Pandas DataFrame if found, or None if the file is missing.
    """
    possible_paths = [
        filepath,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath),
        os.path.join(os.getcwd(), filepath)
    ]
    for p in possible_paths:
        if os.path.exists(p) and os.path.isfile(p):
            try:
                df = pd.read_csv(p)
                return df
            except Exception as e:
                print(f"Error reading CSV at {p}: {e}")
                return None
    return None


# ==============================================================================
# 2. CLEAN DATASET
# ==============================================================================
def clean_dataset(df):
    """
    Performs essential data cleaning:
    - Strips whitespace from column names
    - Strips whitespace from string cell values
    - Validates data types
    """
    if df is None:
        return None
    df_clean = df.copy()
    # Normalize column names
    df_clean.columns = [str(c).strip() for c in df_clean.columns]
    # Clean string entries
    for col in df_clean.select_dtypes(include=['object']):
        df_clean[col] = df_clean[col].astype(str).str.strip()
    return df_clean


# ==============================================================================
# 3. GET DATASET SUMMARY
# ==============================================================================
def get_dataset_summary(df):
    """
    Computes key performance indicators (KPIs) and metadata for the Overview section:
    - Total students and columns
    - Subject score averages (Math, Reading, Writing)
    - Missing values and duplicate count
    - First 6 preview rows and column types
    """
    if df is None:
        return {
            "status": "error",
            "message": f"Dataset file '{CSV_FILE_PATH}' was not found in the application directory."
        }

    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    # Calculate average scores with fallback to available numeric columns
    avg_math = round(float(df['math score'].mean()), 2) if 'math score' in df.columns else (
        round(float(df[numeric_cols[0]].mean()), 2) if len(numeric_cols) > 0 else 0.0
    )
    avg_reading = round(float(df['reading score'].mean()), 2) if 'reading score' in df.columns else (
        round(float(df[numeric_cols[1]].mean()), 2) if len(numeric_cols) > 1 else 0.0
    )
    avg_writing = round(float(df['writing score'].mean()), 2) if 'writing score' in df.columns else (
        round(float(df[numeric_cols[2]].mean()), 2) if len(numeric_cols) > 2 else 0.0
    )

    return {
        "status": "ok",
        "total_students": int(len(df)),
        "total_columns": int(len(df.columns)),
        "avg_math": avg_math,
        "avg_reading": avg_reading,
        "avg_writing": avg_writing,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "columns": df.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "preview": df.head(6).to_dict(orient='records')
    }


# ==============================================================================
# 4. GET NUMERIC COLUMNS
# ==============================================================================
def get_numeric_columns(df):
    """
    Returns a list of all numeric column names in the DataFrame.
    """
    if df is None:
        return []
    return df.select_dtypes(include=[np.number]).columns.tolist()


# ==============================================================================
# 5. GET CATEGORICAL COLUMNS
# ==============================================================================
def get_categorical_columns(df):
    """
    Returns a list of all categorical/object column names in the DataFrame.
    """
    if df is None:
        return []
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


# ==============================================================================
# 6. CREATE HISTOGRAM
# ==============================================================================
def create_histogram(df, column, bins=15, kde=True):
    """
    Generates an educational Histogram with KDE curve and reference lines
    for Mean and Median to teach skewness and central tendency.
    """
    if column not in df.columns or column not in get_numeric_columns(df):
        raise ValueError(f"Column '{column}' is not a valid numeric column.")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    data = df[column].dropna()
    mean_val = data.mean()
    median_val = data.median()

    # Plot histogram with KDE
    sns.histplot(data, bins=bins, kde=kde, color='#3b82f6', edgecolor='#1e40af', alpha=0.65, ax=ax)

    # Reference lines for Mean and Median
    ax.axvline(mean_val, color='#ef4444', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='#10b981', linestyle='-', linewidth=2, label=f'Median: {median_val:.1f}')

    ax.set_title(f"Histogram: Distribution of {column.title()}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(column.title(), fontsize=11)
    ax.set_ylabel("Student Count (Frequency)", fontsize=11)
    ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 7. CREATE BAR CHART
# ==============================================================================
def create_bar_chart(df, x_column, y_column=None):
    """
    Generates a Bar Chart.
    - If y_column is provided: compares the average score across categorical groups with value labels.
    - If y_column is None: shows frequency counts per group.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if y_column and y_column in df.columns:
        grouped = df.groupby(x_column)[y_column].mean().reset_index()
        bars = sns.barplot(data=grouped, x=x_column, y=y_column, hue=x_column, palette='Blues_r', legend=False, ax=ax, edgecolor='#1e293b')
        ax.set_title(f"Bar Chart: Average {y_column.title()} by {x_column.title()}", fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel(f"Average {y_column.title()}", fontsize=11)
        ax.set_ylim(0, 100)
        # Value annotations
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.1f}', (p.get_x() + p.get_width() / 2., height / 2),
                            ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    else:
        sns.countplot(data=df, x=x_column, hue=x_column, palette='Blues_r', legend=False, ax=ax, edgecolor='#1e293b')
        ax.set_title(f"Bar Chart: Frequency of {x_column.title()}", fontsize=13, fontweight='bold', pad=12)
        ax.set_ylabel("Count", fontsize=11)

    ax.set_xlabel(x_column.title(), fontsize=11)
    plt.xticks(rotation=20 if df[x_column].nunique() > 3 else 0, ha='right')
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 8. CREATE LINE CHART
# ==============================================================================
def create_line_chart(df, x_column, y_column=None):
    """
    Generates a Line Chart for ordered or progressive comparisons.
    Groups by the X variable and plots mean score with prominent data markers.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if not y_column or y_column not in df.columns:
        y_column = get_numeric_columns(df)[0]

    # Meaningful sorting for educational stages if parental education is chosen
    edu_order = [
        "some high school", "high school", "some college",
        "associate's degree", "bachelor's degree", "master's degree"
    ]
    grouped = df.groupby(x_column)[y_column].mean()
    if x_column == 'parental level of education':
        grouped = grouped.reindex([e for e in edu_order if e in grouped.index])

    x_vals = [str(x) for x in grouped.index]
    y_vals = grouped.values

    ax.plot(x_vals, y_vals, marker='o', markersize=8, color='#0284c7', linewidth=2.5,
            linestyle='-', label=f"Mean {y_column.title()}")
    ax.fill_between(x_vals, y_vals, alpha=0.12, color='#0284c7')

    # Annotate values above points
    for i, txt in enumerate(y_vals):
        ax.annotate(f'{txt:.1f}', (x_vals[i], y_vals[i] + 1.2), ha='center', fontsize=9, fontweight='bold')

    ax.set_title(f"Line Chart: Progression of {y_column.title()} across {x_column.title()}",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(x_column.title(), fontsize=11)
    ax.set_ylabel(f"Average {y_column.title()}", fontsize=11)
    ax.set_ylim(min(y_vals) - 10 if len(y_vals) > 0 else 0, 100)
    plt.xticks(rotation=25 if len(x_vals) > 3 else 0, ha='right')
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 9. CREATE BOX PLOT
# ==============================================================================
def create_box_plot(df, column, category_column=None):
    """
    Generates a Box Plot (univariate or bivariate by category).
    Shows minimum, Q1 (25th percentile), Median (50th percentile),
    Q3 (75th percentile), Maximum, and potential outliers.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if category_column and category_column in df.columns:
        sns.boxplot(data=df, x=category_column, y=column, hue=category_column, palette='Set2', legend=False, ax=ax,
                    linewidth=1.2, flierprops={'marker': 'o', 'markersize': 5, 'markerfacecolor': '#ef4444'})
        ax.set_title(f"Box Plot: {column.title()} grouped by {category_column.title()}",
                     fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(category_column.title(), fontsize=11)
        plt.xticks(rotation=20 if df[category_column].nunique() > 3 else 0, ha='right')
    else:
        sns.boxplot(y=df[column], color='#93c5fd', ax=ax, width=0.4, linewidth=1.5,
                    flierprops={'marker': 'o', 'markersize': 6, 'markerfacecolor': '#ef4444'})
        ax.set_title(f"Box Plot: Five-Number Summary of {column.title()}",
                     fontsize=13, fontweight='bold', pad=12)
        # Highlight summary numbers
        s = df[column].describe()
        stats_txt = f"Max: {s['max']:.0f}\nQ3: {s['75%']:.0f}\nMedian: {s['50%']:.0f}\nQ1: {s['25%']:.0f}\nMin: {s['min']:.0f}"
        ax.text(0.3, s['50%'], stats_txt, fontsize=10, bbox=dict(facecolor='#f8fafc', edgecolor='#cbd5e1', boxstyle='round'))

    ax.set_ylabel(column.title(), fontsize=11)
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 10. CREATE SCATTER PLOT
# ==============================================================================
def create_scatter_plot(df, x_column, y_column, hue_column=None):
    """
    Generates a Scatter Plot to explore correlation between two continuous variables.
    Computes Pearson's r correlation coefficient and displays a linear trend line.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    valid_df = df[[x_column, y_column]].dropna()
    r = valid_df[x_column].corr(valid_df[y_column])

    # Scatter points
    if hue_column and hue_column in df.columns:
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column,
                        palette='tab10', alpha=0.7, s=55, ax=ax)
        ax.legend(title=hue_column.title(), frameon=True)
    else:
        sns.scatterplot(data=df, x=x_column, y=y_column, color='#2563eb',
                        alpha=0.6, s=55, ax=ax)

    # Linear trendline
    m, b = np.polyfit(valid_df[x_column], valid_df[y_column], 1)
    xseq = np.linspace(valid_df[x_column].min(), valid_df[x_column].max(), num=100)
    ax.plot(xseq, m * xseq + b, color='#dc2626', linestyle='--', linewidth=2, label=f'Trendline (r = {r:.2f})')

    ax.set_title(f"Scatter Plot: {x_column.title()} vs. {y_column.title()} (Pearson r = {r:.2f})",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(x_column.title(), fontsize=11)
    ax.set_ylabel(y_column.title(), fontsize=11)
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 11. CREATE HEATMAP
# ==============================================================================
def create_heatmap(df):
    """
    Generates a Correlation Heatmap of all numeric exam scores.
    Displays Pearson correlation coefficients inside each cell with a diverging colormap.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        raise ValueError("At least 2 numeric columns are required to generate a correlation heatmap.")

    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                linewidths=1.5, linecolor='white', cbar_kws={'label': 'Correlation (r)'}, ax=ax)

    ax.set_title("Multivariate Correlation Heatmap", fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 12. CREATE COUNT PLOT
# ==============================================================================
def create_count_plot(df, column):
    """
    Generates a Count Plot for categorical frequency inspection.
    Includes explicit student headcount labels directly above each bar.
    """
    if column not in df.columns or column not in get_categorical_columns(df):
        raise ValueError(f"Column '{column}' is not a valid categorical column.")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    order = df[column].value_counts().index
    sns.countplot(data=df, x=column, hue=column, order=order, palette='viridis', legend=False, ax=ax, edgecolor='#1e293b')

    # Add count values above each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height + 8),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title(f"Count Plot: Frequency of {column.title()}", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel(column.title(), fontsize=11)
    ax.set_ylabel("Student Count", fontsize=11)
    ax.set_ylim(0, df[column].value_counts().max() * 1.15)
    plt.xticks(rotation=20 if df[column].nunique() > 3 else 0, ha='right')
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 13. CREATE PIE CHART
# ==============================================================================
def create_pie_chart(df, column):
    """
    Generates a clean Donut/Pie Chart for categorical proportions.
    Shows percentage distribution with a central aperture for readability.
    """
    if column not in df.columns or column not in get_categorical_columns(df):
        raise ValueError(f"Column '{column}' is not a valid categorical column.")

    counts = df[column].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    colors = sns.color_palette("pastel")[0:len(counts)]

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2)
    )
    plt.setp(autotexts, size=10, weight="bold")
    ax.set_title(f"Pie Chart: Proportion of {column.title()}", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig_to_base64(fig)


# ==============================================================================
# 14. CREATE PAIRPLOT
# ==============================================================================
def create_pairplot(df, hue=None):
    """
    Generates a Seaborn Pairplot displaying pairwise scatter plots and diagonal
    KDE distributions across all numerical exam scores, optionally colored by group.
    """
    numeric_cols = get_numeric_columns(df)
    subset_cols = list(numeric_cols)
    if hue and hue in df.columns and hue in get_categorical_columns(df):
        subset_cols.append(hue)

    plot_df = df[subset_cols].dropna()
    g = sns.pairplot(plot_df, hue=hue if (hue and hue in plot_df.columns) else None,
                     corner=True, diag_kind='kde', palette='tab10', height=2.2)
    g.fig.suptitle(f"Multivariate Pair Plot{' (by ' + hue.title() + ')' if hue else ''}",
                   fontsize=13, fontweight='bold', y=1.02)
    buf = io.BytesIO()
    g.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(g.fig)
    return f"data:image/png;base64,{img_b64}"


# ==============================================================================
# 15. GET CHART EXPLANATION
# ==============================================================================
def get_chart_explanation(chart_type):
    """
    Returns an educational explanation for beginner students explaining:
    - Purpose
    - When to use
    - What to observe
    - Limitations
    """
    explanations = {
        "histogram": {
            "title": "Histogram (Distribution Analysis)",
            "purpose": "Visualizes the shape, center, spread, and skewness of a single continuous numerical variable.",
            "when_to_use": "Use when examining how exam scores or numerical measurements are spread across values.",
            "what_to_observe": "Notice whether the curve is symmetrical (normal) or skewed left/right. Look for peaks (modes), spread, and outlier gaps.",
            "limitations": "The apparent shape depends heavily on the chosen number of bins; does not display individual observation points."
        },
        "bar": {
            "title": "Bar Chart (Group Comparison)",
            "purpose": "Compares aggregated metrics (such as the average exam score) across distinct categorical groups.",
            "when_to_use": "Use when comparing discrete groups, such as average score by gender, test prep, or lunch status.",
            "what_to_observe": "Compare bar heights to quickly assess relative differences between demographic categories.",
            "limitations": "Shows only summary statistics (like mean); hides internal variance, distribution shape, and outliers within each group."
        },
        "line": {
            "title": "Line Chart (Trend / Ordered Comparison)",
            "purpose": "Displays continuity, trends, or ordered progression across sequential or hierarchical categories.",
            "when_to_use": "Use when categories possess a natural progression, such as levels of parental education from high school to graduate degree.",
            "what_to_observe": "Look for upward or downward slopes, plateaus, and turning points across stages.",
            "limitations": "Improper if categories have no logical order (e.g., race groups), as lines falsely imply continuous progression."
        },
        "box": {
            "title": "Box Plot (Five-Number Summary)",
            "purpose": "Displays median, quartiles (Q1, Q3), interquartile range (IQR), and outliers across distributions.",
            "when_to_use": "Use to compare distributions across groups and identify extreme outlier scores.",
            "what_to_observe": "Observe box length (spread/IQR), median line position (skewness), and individual outlier dots beyond whiskers.",
            "limitations": "Can obscure multi-modal distributions (bimodal distributions may look similar to unimodal in a standard box)."
        },
        "scatter": {
            "title": "Scatter Plot (Bivariate Relationship)",
            "purpose": "Investigates correlation, linearity, and clustering between two continuous numerical variables.",
            "when_to_use": "Use when exploring how one exam score relates to another (e.g., math score vs. reading score).",
            "what_to_observe": "Check if points form an upward slope (positive correlation) or downward slope, and note how tightly clustered points are.",
            "limitations": "Susceptible to overplotting when handling thousands of overlapping points; cannot establish causation."
        },
        "heatmap": {
            "title": "Correlation Heatmap (Multivariate Matrix)",
            "purpose": "Summarizes pairwise Pearson correlation coefficients (r) across all numeric variables using color gradients.",
            "when_to_use": "Use during initial EDA to identify which variables move together or suffer from multicollinearity.",
            "what_to_observe": "Identify strong positive values (+0.7 to +1.0) or negative values (-0.7 to -1.0). Reading and writing scores typically exhibit high correlation.",
            "limitations": "Only measures linear relationships; non-linear associations may register close to zero."
        },
        "count": {
            "title": "Count Plot (Categorical Frequencies)",
            "purpose": "Displays the absolute frequency (headcount) of observations in each category of a qualitative feature.",
            "when_to_use": "Use to detect class imbalances in demographic factors like gender, test prep participation, or lunch type.",
            "what_to_observe": "Identify dominant categories vs. underrepresented groups that might influence statistical conclusions.",
            "limitations": "Shows only counts; cannot show relationships with performance scores without pairing."
        },
        "pie": {
            "title": "Pie / Donut Chart (Proportion Analysis)",
            "purpose": "Visualizes part-to-whole categorical percentages where all slices sum to 100%.",
            "when_to_use": "Use only when you have 2 to 5 distinct categories, such as gender or lunch type.",
            "what_to_observe": "Check the relative share and dominance of each category in the dataset.",
            "limitations": "Human vision struggles to accurately compare angles and slice areas, especially with more than 4 slices."
        },
        "pairplot": {
            "title": "Pair Plot (Multivariate Exploration)",
            "purpose": "Plots a matrix of all pairwise scatter plots and univariate diagonal distributions in one comprehensive view.",
            "when_to_use": "Use at the start of analysis to simultaneously inspect all bivariate pairs and their grouping patterns.",
            "what_to_observe": "Scan off-diagonal plots for linear relationships and diagonal plots for subgroup distribution shifts.",
            "limitations": "Computationally intensive on large datasets and high dimensional feature sets."
        }
    }
    return explanations.get(chart_type, {
        "title": "Educational Chart",
        "purpose": "Visualizes exploratory patterns in the dataset.",
        "when_to_use": "Use during exploratory data analysis.",
        "what_to_observe": "Observe central tendency, spread, and relationships.",
        "limitations": "Ensure appropriate chart choice for the specific data type."
    })


# ==============================================================================
# 16. GENERATE MATPLOTLIB CODE
# ==============================================================================
def generate_matplotlib_code(chart_type, x_column="math score", y_column="reading score", hue_column=None):
    """
    Returns clean, beginner-friendly Matplotlib Python code demonstrating
    how to construct the given chart from scratch.
    """
    code_templates = {
        "histogram": f"""# Matplotlib: Histogram
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(df['{x_column}'], bins=15, color='#3b82f6', edgecolor='black', alpha=0.7)
plt.axvline(df['{x_column}'].mean(), color='red', linestyle='--', label='Mean')
plt.title("Distribution of {x_column.title()}", fontsize=14)
plt.xlabel("{x_column.title()}")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()""",

        "bar": f"""# Matplotlib: Bar Chart
import matplotlib.pyplot as plt

# Step 1: Compute group means with Pandas
grouped = df.groupby('{x_column}')['{y_column}'].mean()

# Step 2: Plot bars manually
plt.figure(figsize=(8, 5))
bars = plt.bar(grouped.index, grouped.values, color='#3b82f6', edgecolor='black')
plt.title("Average {y_column.title()} by {x_column.title()}", fontsize=14)
plt.xlabel("{x_column.title()}")
plt.ylabel("Average {y_column.title()}")
plt.xticks(rotation=20)

# Step 3: Add value labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{{yval:.1f}}", ha='center')
plt.show()""",

        "line": f"""# Matplotlib: Line Chart
import matplotlib.pyplot as plt

# Group and calculate mean
trend = df.groupby('{x_column}')['{y_column}'].mean()

plt.figure(figsize=(8, 5))
plt.plot(trend.index, trend.values, marker='o', color='#0284c7', linewidth=2.5)
plt.title("Trend: {y_column.title()} across {x_column.title()}", fontsize=14)
plt.xlabel("{x_column.title()}")
plt.ylabel("Average {y_column.title()}")
plt.xticks(rotation=25)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()""",

        "box": f"""# Matplotlib: Box Plot
import matplotlib.pyplot as plt

# Prepare grouped lists manually for Matplotlib
categories = df['{x_column}'].unique()
data_per_cat = [df[df['{x_column}'] == cat]['{y_column}'].dropna() for cat in categories]

plt.figure(figsize=(8, 5))
plt.boxplot(data_per_cat, labels=categories, patch_artist=True)
plt.title("{y_column.title()} by {x_column.title()}", fontsize=14)
plt.xlabel("{x_column.title()}")
plt.ylabel("{y_column.title()}")
plt.show()""",

        "scatter": f"""# Matplotlib: Scatter Plot
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 5))
plt.scatter(df['{x_column}'], df['{y_column}'], color='#2563eb', alpha=0.6)

# Add trend line using NumPy polyfit
m, b = np.polyfit(df['{x_column}'], df['{y_column}'], 1)
plt.plot(df['{x_column}'], m*df['{x_column}'] + b, color='red', linestyle='--')

plt.title("{x_column.title()} vs. {y_column.title()}", fontsize=14)
plt.xlabel("{x_column.title()}")
plt.ylabel("{y_column.title()}")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()""",

        "heatmap": """# Matplotlib: Correlation Heatmap
import matplotlib.pyplot as plt

numeric_cols = ['math score', 'reading score', 'writing score']
corr = df[numeric_cols].corr()

plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr)), corr.columns, rotation=20)
plt.yticks(range(len(corr)), corr.columns)

# Matplotlib requires nested loops to annotate correlation text
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')

plt.title("Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.show()""",

        "count": f"""# Matplotlib: Count Plot
import matplotlib.pyplot as plt

counts = df['{x_column}'].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(counts.index, counts.values, color='#4f46e5', edgecolor='black')
plt.title("Frequency of {x_column.title()}", fontsize=14)
plt.xlabel("{x_column.title()}")
plt.ylabel("Count")
plt.xticks(rotation=20)
plt.show()""",

        "pie": f"""# Matplotlib: Pie / Donut Chart
import matplotlib.pyplot as plt

counts = df['{x_column}'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%',
        startangle=140, wedgeprops=dict(width=0.6, edgecolor='white'))
plt.title("Proportion of {x_column.title()}", fontsize=14)
plt.show()""",

        "pairplot": """# Matplotlib: Pair Plot Matrix
import matplotlib.pyplot as plt

cols = ['math score', 'reading score', 'writing score']
fig, axes = plt.subplots(len(cols), len(cols), figsize=(8, 8))

# Matplotlib requires manual nested subplot grid programming
for i, col1 in enumerate(cols):
    for j, col2 in enumerate(cols):
        if i == j:
            axes[i, j].hist(df[col1], bins=15, color='#3b82f6')
        else:
            axes[i, j].scatter(df[col2], df[col1], alpha=0.4, s=10)
        if j == 0: axes[i, j].set_ylabel(col1)
        if i == len(cols)-1: axes[i, j].set_xlabel(col2)

plt.tight_layout()
plt.show()"""
    }
    return code_templates.get(chart_type, "# Matplotlib code not available for this chart type.")


# ==============================================================================
# 17. GENERATE SEABORN CODE
# ==============================================================================
def generate_seaborn_code(chart_type, x_column="math score", y_column="reading score", hue_column=None):
    """
    Returns clean, beginner-friendly Seaborn Python code demonstrating
    how to construct the given chart with modern statistical idioms.
    """
    hue_param = f", hue='{hue_column}'" if hue_column else ""
    code_templates = {
        "histogram": f"""# Seaborn: Histogram with KDE
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
# One line handles bins and smooth density estimation!
sns.histplot(data=df, x='{x_column}', bins=15, kde=True, color='#2563eb')
plt.title("Distribution of {x_column.title()} with KDE", fontsize=14)
plt.xlabel("{x_column.title()}")
plt.ylabel("Student Count")
plt.show()""",

        "bar": f"""# Seaborn: Bar Chart
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
# Seaborn automatically computes means and handles grouping directly!
sns.barplot(data=df, x='{x_column}', y='{y_column}'{hue_param}, palette='Blues_d', ci=None)
plt.title("Average {y_column.title()} by {x_column.title()}", fontsize=14)
plt.xlabel("{x_column.title()}")
plt.ylabel("Average {y_column.title()}")
plt.xticks(rotation=20)
plt.show()""",

        "line": f"""# Seaborn: Line Chart
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
# Automatically computes group means and confidence intervals
sns.lineplot(data=df, x='{x_column}', y='{y_column}', marker='o', ci=None, color='#0284c7')
plt.title("Average {y_column.title()} across {x_column.title()}", fontsize=14)
plt.xticks(rotation=25)
plt.show()""",

        "box": f"""# Seaborn: Box Plot
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
# Directly pass DataFrame columns without manual reshaping
sns.boxplot(data=df, x='{x_column}', y='{y_column}'{hue_param}, palette='Set2')
plt.title("{y_column.title()} Distribution by {x_column.title()}", fontsize=14)
plt.show()""",

        "scatter": f"""# Seaborn: Scatter Plot with Hue
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
# Easily color points by demographic categories using hue
sns.scatterplot(data=df, x='{x_column}', y='{y_column}'{hue_param}, alpha=0.7, palette='tab10')
plt.title("{x_column.title()} vs. {y_column.title()}", fontsize=14)
plt.show()""",

        "heatmap": """# Seaborn: Correlation Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

numeric_cols = ['math score', 'reading score', 'writing score']
plt.figure(figsize=(6, 5))
# annot=True automatically displays r-values inside each cell!
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1)
plt.title("Correlation Heatmap (Seaborn)", fontsize=14)
plt.tight_layout()
plt.show()""",

        "count": f"""# Seaborn: Count Plot
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
# Automatically counts occurrences per category
sns.countplot(data=df, x='{x_column}', palette='viridis')
plt.title("Frequency of {x_column.title()}", fontsize=14)
plt.xticks(rotation=20)
plt.show()""",

        "pie": """# Note: Seaborn does NOT have a dedicated pie chart function!
# In statistical data science, pie charts are generally discouraged
# because angle comparisons are imprecise. Instead, use sns.countplot()
# or Matplotlib's plt.pie().""",

        "pairplot": f"""# Seaborn: Pair Plot Matrix
import seaborn as sns
import matplotlib.pyplot as plt

# A full multivariate scatter & KDE matrix in a single function call!
sns.pairplot(df[['math score', 'reading score', 'writing score'{", '" + hue_column + "'" if hue_column else ""}]],
             hue={'\"' + hue_column + '\"' if hue_column else 'None'}, corner=True, diag_kind='kde')
plt.show()"""
    }
    return code_templates.get(chart_type, "# Seaborn code not available for this chart type.")


# ==============================================================================
# 18. CREATE DASHBOARD HTML
# ==============================================================================
def create_dashboard_html():
    """
    Returns the complete single-file HTML document containing embedded CSS,
    classroom teaching sections, educational interactive studios, side-by-side
    Matplotlib vs Seaborn laboratory, graph comparison guide, and vanilla JavaScript.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Exploratory Data Analysis (EDA) Interactive Classroom Dashboard</title>
  <style>
    /* =========================================================================
       CLASSROOM TEACHER DASHBOARD STYLES (Clean, Neat, Academic)
       ========================================================================= */
    :root {
      --primary: #1e40af;
      --primary-light: #3b82f6;
      --primary-bg: #eff6ff;
      --secondary: #0f766e;
      --text-main: #0f172a;
      --text-muted: #475569;
      --bg-page: #f8fafc;
      --bg-card: #ffffff;
      --border-color: #e2e8f0;
      --border-dark: #cbd5e1;
      --accent-green: #059669;
      --accent-red: #dc2626;
      --accent-amber: #d97706;
      --code-bg: #1e293b;
      --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
      --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
      --font-main: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-main);
      background-color: var(--bg-page);
      color: var(--text-main);
      line-height: 1.5;
      padding-bottom: 60px;
    }

    /* TOP HEADER & NAVBAR */
    header {
      background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
      color: #ffffff;
      padding: 24px 32px;
      border-bottom: 3px solid #60a5fa;
      box-shadow: var(--shadow-md);
    }
    .header-content {
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }
    .header-title h1 {
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .header-title p {
      font-size: 0.95rem;
      color: #bfdbfe;
      margin-top: 4px;
    }
    .header-badge {
      background: rgba(255, 255, 255, 0.15);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 500;
      border: 1px solid rgba(255, 255, 255, 0.25);
    }

    /* NAVIGATION TABS */
    nav.subnav {
      background: #ffffff;
      border-bottom: 1px solid var(--border-color);
      position: sticky;
      top: 0;
      z-index: 50;
      box-shadow: var(--shadow-sm);
    }
    .nav-container {
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      overflow-x: auto;
      padding: 0 16px;
    }
    .nav-btn {
      padding: 14px 18px;
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--text-muted);
      border: none;
      background: none;
      cursor: pointer;
      white-space: nowrap;
      border-bottom: 3px solid transparent;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .nav-btn:hover { color: var(--primary); background: #f1f5f9; }
    .nav-btn.active {
      color: var(--primary);
      border-bottom-color: var(--primary);
      background: var(--primary-bg);
    }

    /* CONTAINER & SECTIONS */
    .container {
      max-width: 1200px;
      margin: 28px auto 0;
      padding: 0 20px;
    }
    .section-block {
      display: none;
    }
    .section-block.active {
      display: block;
    }

    /* ALERT BANNERS */
    .alert-banner {
      padding: 16px 20px;
      border-radius: 8px;
      margin-bottom: 24px;
      display: flex;
      align-items: flex-start;
      gap: 12px;
      border-left: 5px solid;
    }
    .alert-error {
      background-color: #fef2f2;
      border-color: var(--accent-red);
      color: #991b1b;
    }
    .alert-info {
      background-color: #f0fdf4;
      border-color: var(--accent-green);
      color: #166534;
    }

    /* KPI CARDS GRID */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }
    .kpi-card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px 18px;
      box-shadow: var(--shadow-sm);
      transition: transform 0.1s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-label {
      font-size: 0.78rem;
      text-transform: uppercase;
      font-weight: 700;
      color: var(--text-muted);
      letter-spacing: 0.05em;
    }
    .kpi-value {
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--primary);
      margin-top: 6px;
    }
    .kpi-sub {
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 4px;
    }

    /* WHITE CARDS */
    .card {
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: var(--shadow-sm);
    }
    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .card-subtitle {
      font-size: 0.9rem;
      color: var(--text-muted);
      margin-bottom: 18px;
    }

    /* TEACHING PILLARS GRID */
    .concept-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
      margin-bottom: 24px;
    }
    .concept-box {
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 18px;
      background: #fafafa;
    }
    .concept-box h3 {
      font-size: 1rem;
      font-weight: 700;
      color: var(--primary);
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .concept-box p {
      font-size: 0.88rem;
      color: var(--text-muted);
      margin-bottom: 10px;
    }
    .concept-box ul {
      padding-left: 20px;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    .concept-box li { margin-bottom: 4px; }

    /* CONTROLS BAR (Dropdowns, Buttons) */
    .controls-panel {
      background: #f1f5f9;
      border: 1px solid var(--border-dark);
      border-radius: 8px;
      padding: 16px 20px;
      margin-bottom: 20px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 16px;
    }
    .control-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .control-group label {
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
    }
    select, button {
      font-family: inherit;
      font-size: 0.9rem;
      padding: 8px 12px;
      border-radius: 6px;
      border: 1px solid var(--border-dark);
      background: #ffffff;
      color: var(--text-main);
      outline: none;
    }
    select:focus {
      border-color: var(--primary-light);
      box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
    }
    .btn-primary {
      background: var(--primary);
      color: #ffffff;
      border: 1px solid #1e3a8a;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s ease;
      align-self: flex-end;
    }
    .btn-primary:hover { background: #1e3a8a; }
    .btn-secondary {
      background: #ffffff;
      color: var(--text-main);
      border: 1px solid var(--border-dark);
      font-weight: 600;
      cursor: pointer;
    }
    .btn-secondary:hover { background: #f8fafc; }
    .preset-group {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
      align-items: center;
    }
    .preset-label {
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--text-muted);
    }
    .btn-preset {
      font-size: 0.8rem;
      padding: 4px 10px;
      border-radius: 9999px;
      background: #e2e8f0;
      border: 1px solid #cbd5e1;
      cursor: pointer;
      transition: all 0.1s ease;
    }
    .btn-preset:hover {
      background: var(--primary-light);
      color: #ffffff;
      border-color: var(--primary);
    }

    /* CHART OUTPUT & EXPLANATION SPLIT */
    .studio-layout {
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 24px;
      align-items: start;
    }
    @media (max-width: 900px) {
      .studio-layout { grid-template-columns: 1fr; }
    }
    .chart-container {
      background: #ffffff;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 420px;
      box-shadow: var(--shadow-sm);
    }
    .chart-container img {
      max-width: 100%;
      height: auto;
      border-radius: 4px;
    }
    .loading-spinner {
      display: none;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      color: var(--text-muted);
    }
    .spinner {
      width: 38px;
      height: 38px;
      border: 4px solid #e2e8f0;
      border-top-color: var(--primary-light);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* TEACHER EXPLANATION CARD */
    .teach-card {
      background: #ffffff;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 20px;
      box-shadow: var(--shadow-sm);
    }
    .teach-title {
      font-size: 1.1rem;
      font-weight: 700;
      color: var(--primary);
      margin-bottom: 12px;
      border-bottom: 2px solid var(--primary-bg);
      padding-bottom: 8px;
    }
    .teach-item {
      margin-bottom: 14px;
    }
    .teach-item-label {
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 3px;
    }
    .teach-item-text {
      font-size: 0.88rem;
      color: var(--text-main);
      line-height: 1.45;
    }

    /* CODE TOGGLE & DISPLAY */
    .code-toggle-bar {
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid var(--border-color);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .code-section {
      display: none;
      margin-top: 14px;
    }
    .code-section.visible { display: block; }
    .code-tabs {
      display: flex;
      gap: 6px;
      margin-bottom: 6px;
    }
    .code-tab-btn {
      font-size: 0.8rem;
      padding: 4px 10px;
      background: #e2e8f0;
      border: 1px solid #cbd5e1;
      border-radius: 4px;
      cursor: pointer;
    }
    .code-tab-btn.active {
      background: var(--code-bg);
      color: #ffffff;
      border-color: var(--code-bg);
    }
    pre.code-block {
      background: var(--code-bg);
      color: #f8fafc;
      padding: 14px;
      border-radius: 6px;
      font-family: var(--font-mono);
      font-size: 0.82rem;
      overflow-x: auto;
      white-space: pre;
      line-height: 1.4;
      border: 1px solid #334155;
    }

    /* DATASET TABLE PREVIEW */
    .table-wrapper {
      overflow-x: auto;
      margin-top: 14px;
      border: 1px solid var(--border-color);
      border-radius: 6px;
    }
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      text-align: left;
    }
    table.data-table th {
      background: #f1f5f9;
      color: var(--text-main);
      font-weight: 600;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border-color);
    }
    table.data-table td {
      padding: 9px 14px;
      border-bottom: 1px solid var(--border-color);
      color: var(--text-muted);
    }
    table.data-table tr:last-child td { border-bottom: none; }
    table.data-table tr:hover { background: #f8fafc; }

    /* SIDE BY SIDE CODE LABORATORY */
    .lab-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-top: 16px;
    }
    @media (max-width: 850px) {
      .lab-grid { grid-template-columns: 1fr; }
    }
    .lab-col {
      border: 1px solid var(--border-color);
      border-radius: 8px;
      overflow: hidden;
      background: #ffffff;
    }
    .lab-header {
      padding: 12px 16px;
      font-weight: 700;
      font-size: 0.95rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border-color);
    }
    .lab-header.mpl { background: #eff6ff; color: #1d4ed8; }
    .lab-header.sns { background: #f0fdf4; color: #15803d; }
    .lab-body { padding: 14px; }
    .lab-body pre { margin: 0; max-height: 380px; }

    /* COMPARISON REFERENCE TABLE */
    table.ref-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      font-size: 0.88rem;
    }
    table.ref-table th {
      background: #1e3a8a;
      color: #ffffff;
      padding: 12px 14px;
      font-weight: 600;
      text-align: left;
      border: 1px solid #1e40af;
    }
    table.ref-table td {
      padding: 12px 14px;
      border: 1px solid var(--border-color);
      vertical-align: top;
    }
    table.ref-table tr:nth-child(even) { background: #f8fafc; }
    table.ref-table tr:hover { background: #f1f5f9; }
    .tag-badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
      background: #e2e8f0;
      color: #334155;
    }

    footer {
      text-align: center;
      margin-top: 48px;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
  </style>
</head>
<body>

  <!-- HEADER -->
  <header>
    <div class="header-content">
      <div class="header-title">
        <h1>Exploratory Data Analysis (EDA) Interactive Dashboard</h1>
        <p>A classroom teaching laboratory for mastering Data Science, Matplotlib, and Seaborn</p>
      </div>
      <div class="header-badge" id="datasetBadge">
        Dataset: StudentsPerformance.csv
      </div>
    </div>
  </header>

  <!-- NAVIGATION TABS -->
  <nav class="subnav">
    <div class="nav-container">
      <button class="nav-btn active" onclick="switchSection('overview')">📊 Overview & Dataset</button>
      <button class="nav-btn" onclick="switchSection('what-is-eda')">🎓 What is EDA?</button>
      <button class="nav-btn" onclick="switchSection('univariate')">📈 Univariate Analysis</button>
      <button class="nav-btn" onclick="switchSection('bivariate')">🔍 Bivariate Analysis</button>
      <button class="nav-btn" onclick="switchSection('multivariate')">🌐 Multivariate Analysis</button>
      <button class="nav-btn" onclick="switchSection('mpl-vs-sns')">⚖️ Matplotlib vs. Seaborn</button>
      <button class="nav-btn" onclick="switchSection('comparison-table')">📚 Graph Comparison Guide</button>
    </div>
  </nav>

  <!-- MAIN CONTAINER -->
  <div class="container">

    <!-- ERROR NOTIFICATION BANNER (IF CSV MISSING) -->
    <div id="missingFileAlert" class="alert-banner alert-error" style="display: none;">
      <div>
        <strong>File Missing:</strong> The dataset file <code>StudentsPerformance.csv</code> was not found in the application directory.
        <div style="margin-top: 6px; font-size: 0.88rem;">
          Please download the dataset from Kaggle (<code>Students' Performance in Exams</code>) and place the CSV file in the same folder as <code>eda_dashboard.py</code>.
        </div>
      </div>
    </div>

    <!-- =====================================================================
         SECTION 1: OVERVIEW & DATASET
         ===================================================================== -->
    <div id="section-overview" class="section-block active">
      <!-- SUMMARY METRIC CARDS -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Total Students</div>
          <div class="kpi-value" id="kpiStudents">-</div>
          <div class="kpi-sub">Exam candidates</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Total Columns</div>
          <div class="kpi-value" id="kpiColumns">-</div>
          <div class="kpi-sub">Features analyzed</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Avg Math Score</div>
          <div class="kpi-value" id="kpiMath">-</div>
          <div class="kpi-sub">Scale: 0 - 100</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Avg Reading Score</div>
          <div class="kpi-value" id="kpiReading">-</div>
          <div class="kpi-sub">Scale: 0 - 100</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Avg Writing Score</div>
          <div class="kpi-value" id="kpiWriting">-</div>
          <div class="kpi-sub">Scale: 0 - 100</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Missing Values</div>
          <div class="kpi-value" id="kpiMissing" style="color: var(--accent-green);">-</div>
          <div class="kpi-sub">Clean dataset</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Duplicate Rows</div>
          <div class="kpi-value" id="kpiDuplicates" style="color: var(--accent-green);">-</div>
          <div class="kpi-sub">Unique records</div>
        </div>
      </div>

      <!-- DATASET PREVIEW -->
      <div class="card">
        <div class="card-title">📄 Dataset Preview & Data Types</div>
        <div class="card-subtitle">
          Displaying the first 6 observations from the Students Performance dataset.
        </div>
        <div class="table-wrapper">
          <table class="data-table" id="previewTable">
            <thead>
              <tr id="previewTableHead"><th>Loading table...</th></tr>
            </thead>
            <tbody id="previewTableBody">
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- =====================================================================
         SECTION 2: WHAT IS EDA?
         ===================================================================== -->
    <div id="section-what-is-eda" class="section-block">
      <div class="card">
        <div class="card-title">📖 What is Exploratory Data Analysis (EDA)?</div>
        <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 20px;">
          Pioneered by mathematician <strong>John Tukey in 1977</strong>, Exploratory Data Analysis is the essential
          first step in data science. Rather than jumping straight into complex machine learning models or formal hypothesis tests,
          EDA emphasizes <em>listening to the data</em> through visual techniques, uncovering anomalies, assessing distributions,
          and forming grounded hypotheses.
        </p>

        <div class="concept-grid">
          <div class="concept-box">
            <h3>1. Univariate Analysis</h3>
            <p>Explores a <strong>single variable</strong> in isolation without considering relationships with other features.</p>
            <ul>
              <li><strong>Continuous features:</strong> Examine shape, center (mean/median), spread (variance/IQR), and skewness using <em>Histograms</em> and <em>Box Plots</em>.</li>
              <li><strong>Categorical features:</strong> Count frequencies and proportions using <em>Count Plots</em> and <em>Pie Charts</em>.</li>
              <li><strong>Core Question:</strong> <em>"What does the score distribution look like across our students?"</em></li>
            </ul>
          </div>

          <div class="concept-box">
            <h3>2. Bivariate Analysis</h3>
            <p>Investigates the <strong>relationship between two variables</strong> to determine how one behaves in relation to another.</p>
            <ul>
              <li><strong>Numeric vs. Numeric:</strong> Uncover correlation, linearity, and clusters using <em>Scatter Plots</em>.</li>
              <li><strong>Categorical vs. Numeric:</strong> Compare subgroup means and spreads using <em>Bar Charts</em> and <em>Grouped Box Plots</em>.</li>
              <li><strong>Core Question:</strong> <em>"Do students who completed test preparation score higher in math?"</em></li>
            </ul>
          </div>

          <div class="concept-box">
            <h3>3. Multivariate Analysis</h3>
            <p>Analyzes <strong>three or more variables simultaneously</strong> to reveal complex interactions, confounding factors, and groupings.</p>
            <ul>
              <li><strong>Correlation Matrices:</strong> Identify multi-variable linear correlations using <em>Heatmaps</em>.</li>
              <li><strong>Pairwise Matrices:</strong> Scan all variable pairs colored by category with <em>Pair Plots</em>.</li>
              <li><strong>Core Question:</strong> <em>"How do gender, lunch type, and parental education jointly impact reading and writing?"</em></li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <!-- =====================================================================
         SECTION 3: UNIVARIATE ANALYSIS STUDIO
         ===================================================================== -->
    <div id="section-univariate" class="section-block">
      <div class="card">
        <div class="card-title">📈 Univariate Analysis Studio</div>
        <div class="card-subtitle">
          Select a single numeric or categorical variable to explore its individual distribution, central tendency, and spread.
        </div>

        <!-- CONTROLS -->
        <div class="controls-panel">
          <div class="control-group">
            <label for="uniColSelect">Select Column</label>
            <select id="uniColSelect" onchange="updateUniChartOptions()">
              <!-- Populated by JS -->
            </select>
          </div>

          <div class="control-group">
            <label for="uniTypeSelect">Chart Type</label>
            <select id="uniTypeSelect">
              <option value="histogram">Histogram (Numeric)</option>
              <option value="box">Box Plot (Numeric)</option>
              <option value="count">Count Plot (Categorical)</option>
              <option value="pie">Pie Chart (Categorical)</option>
            </select>
          </div>

          <button class="btn-primary" onclick="renderUnivariateChart()">Generate Chart</button>
        </div>

        <!-- STUDIO LAYOUT -->
        <div class="studio-layout">
          <!-- CHART DISPLAY -->
          <div class="chart-container" id="uniChartBox">
            <div class="loading-spinner" id="uniSpinner">
              <div class="spinner"></div>
              <span>Rendering chart with Matplotlib/Seaborn...</span>
            </div>
            <img id="uniChartImg" src="" alt="Univariate Chart" style="display: none;">
          </div>

          <!-- TEACHER EXPLANATION & CODE -->
          <div class="teach-card">
            <div class="teach-title" id="uniTeachTitle">Histogram Analysis</div>
            <div class="teach-item">
              <div class="teach-item-label">Purpose</div>
              <div class="teach-item-text" id="uniTeachPurpose">Loading purpose...</div>
            </div>
            <div class="teach-item">
              <div class="teach-item-label">When to Use</div>
              <div class="teach-item-text" id="uniTeachWhen">Loading usage...</div>
            </div>
            <div class="teach-item">
              <div class="teach-item-label">What to Observe</div>
              <div class="teach-item-text" id="uniTeachObserve">Loading observations...</div>
            </div>
            <div class="teach-item">
              <div class="teach-item-label">Limitations</div>
              <div class="teach-item-text" id="uniTeachLimits">Loading limitations...</div>
            </div>

            <!-- CODE TOGGLE -->
            <div class="code-toggle-bar">
              <span style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">Python Code:</span>
              <button class="btn-secondary" onclick="toggleCode('uniCodeSection')">Toggle Code</button>
            </div>
            <div id="uniCodeSection" class="code-section">
              <div class="code-tabs">
                <button class="code-tab-btn active" onclick="switchCodeTab('uni', 'mpl')">Matplotlib</button>
                <button class="code-tab-btn" onclick="switchCodeTab('uni', 'sns')">Seaborn</button>
              </div>
              <pre class="code-block" id="uniCodeMpl"></pre>
              <pre class="code-block" id="uniCodeSns" style="display: none;"></pre>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- =====================================================================
         SECTION 4: BIVARIATE ANALYSIS STUDIO
         ===================================================================== -->
    <div id="section-bivariate" class="section-block">
      <div class="card">
        <div class="card-title">🔍 Bivariate Analysis Studio</div>
        <div class="card-subtitle">
          Examine relationships between two variables: compare group averages, evaluate correlations, or track progression trends.
        </div>

        <!-- QUICK PRESET BUTTONS -->
        <div class="preset-group">
          <span class="preset-label">Classroom Presets:</span>
          <button class="btn-preset" onclick="loadBivariatePreset('gender', 'math score', 'bar')">Math score by Gender (Bar)</button>
          <button class="btn-preset" onclick="loadBivariatePreset('math score', 'reading score', 'scatter')">Math vs Reading (Scatter)</button>
          <button class="btn-preset" onclick="loadBivariatePreset('lunch', 'writing score', 'box')">Writing by Lunch type (Box)</button>
          <button class="btn-preset" onclick="loadBivariatePreset('parental level of education', 'math score', 'line')">Math by Parental Education (Line)</button>
        </div>

        <!-- CONTROLS -->
        <div class="controls-panel" style="margin-top: 14px;">
          <div class="control-group">
            <label for="biXSelect">X Variable</label>
            <select id="biXSelect"></select>
          </div>
          <div class="control-group">
            <label for="biYSelect">Y Variable</label>
            <select id="biYSelect"></select>
          </div>
          <div class="control-group">
            <label for="biTypeSelect">Chart Type</label>
            <select id="biTypeSelect">
              <option value="bar">Bar Chart (Group Mean Comparison)</option>
              <option value="scatter">Scatter Plot (Two Numeric Variables)</option>
              <option value="box">Box Plot by Category</option>
              <option value="line">Line Chart (Ordered Progression)</option>
            </select>
          </div>
          <button class="btn-primary" onclick="renderBivariateChart()">Generate Chart</button>
        </div>

        <!-- STUDIO LAYOUT -->
        <div class="studio-layout">
          <div class="chart-container" id="biChartBox">
            <div class="loading-spinner" id="biSpinner">
              <div class="spinner"></div>
              <span>Rendering bivariate chart...</span>
            </div>
            <img id="biChartImg" src="" alt="Bivariate Chart" style="display: none;">
          </div>

          <div class="teach-card">
            <div class="teach-title" id="biTeachTitle">Bivariate Analysis</div>
            <div class="teach-item">
              <div class="teach-item-label">Purpose</div>
              <div class="teach-item-text" id="biTeachPurpose">Loading purpose...</div>
            </div>
            <div class="teach-item">
              <div class="teach-item-label">When to Use</div>
              <div class="teach-item-text" id="biTeachWhen">Loading usage...</div>
            </div>
            <div class="teach-item">
              <div class="teach-item-label">What to Observe</div>
              <div class="teach-item-text" id="biTeachObserve">Loading observations...</div>
            </div>
            <div class="teach-item">
              <div class="teach-item-label">Limitations</div>
              <div class="teach-item-text" id="biTeachLimits">Loading limitations...</div>
            </div>

            <!-- CODE TOGGLE -->
            <div class="code-toggle-bar">
              <span style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">Python Code:</span>
              <button class="btn-secondary" onclick="toggleCode('biCodeSection')">Toggle Code</button>
            </div>
            <div id="biCodeSection" class="code-section">
              <div class="code-tabs">
                <button class="code-tab-btn active" onclick="switchCodeTab('bi', 'mpl')">Matplotlib</button>
                <button class="code-tab-btn" onclick="switchCodeTab('bi', 'sns')">Seaborn</button>
              </div>
              <pre class="code-block" id="biCodeMpl"></pre>
              <pre class="code-block" id="biCodeSns" style="display: none;"></pre>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- =====================================================================
         SECTION 5: MULTIVARIATE ANALYSIS STUDIO
         ===================================================================== -->
    <div id="section-multivariate" class="section-block">
      <div class="card">
        <div class="card-title">🌐 Multivariate Analysis Studio</div>
        <div class="card-subtitle">
          Uncover complex relationships across 3 or more variables simultaneously using correlation matrices and pair plots.
        </div>

        <div class="controls-panel">
          <div class="control-group">
            <label for="multiTypeSelect">Multivariate Method</label>
            <select id="multiTypeSelect" onchange="updateMultiControls()">
              <option value="heatmap">Correlation Heatmap (All Exam Scores)</option>
              <option value="pairplot">Pair Plot Matrix (Scores with Hue Grouping)</option>
            </select>
          </div>
          <div class="control-group" id="multiHueGroup">
            <label for="multiHueSelect">Color Grouping (Hue)</label>
            <select id="multiHueSelect">
              <option value="none">None</option>
              <option value="gender">Gender</option>
              <option value="lunch">Lunch Type</option>
              <option value="test preparation course">Test Prep Course</option>
            </select>
          </div>
          <button class="btn-primary" onclick="renderMultivariateChart()">Generate Chart</button>
        </div>

        <div class="studio-layout">
          <div class="chart-container" id="multiChartBox">
            <div class="loading-spinner" id="multiSpinner">
              <div class="spinner"></div>
              <span>Rendering multivariate matrix (may take a moment)...</span>
            </div>
            <img id="multiChartImg" src="" alt="Multivariate Chart" style="display: none;">
          </div>

          <div class="teach-card">
            <div class="teach-title" id="multiTeachTitle">Multivariate Analysis</div>
            <div class="teach-item">
              <div class="teach-item-label">Purpose</div>
              <div class="teach-item-text" id="multiTeachPurpose">Loading purpose...</div>
            </div>
            <div class="teach-item">
              <div class="teach-item-label">When to Use</div>
              <div class="teach-item-text" id="multiTeachWhen">Loading usage...</div>
            </div>
            <div class="teach-item">
              <div class="teach-item-label">What to Observe</div>
              <div class="teach-item-text" id="multiTeachObserve">Loading observations...</div>
            </div>
            <div class="teach-item">
              <div class="teach-item-label">Limitations</div>
              <div class="teach-item-text" id="multiTeachLimits">Loading limitations...</div>
            </div>

            <!-- CODE TOGGLE -->
            <div class="code-toggle-bar">
              <span style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted);">Python Code:</span>
              <button class="btn-secondary" onclick="toggleCode('multiCodeSection')">Toggle Code</button>
            </div>
            <div id="multiCodeSection" class="code-section">
              <div class="code-tabs">
                <button class="code-tab-btn active" onclick="switchCodeTab('multi', 'mpl')">Matplotlib</button>
                <button class="code-tab-btn" onclick="switchCodeTab('multi', 'sns')">Seaborn</button>
              </div>
              <pre class="code-block" id="multiCodeMpl"></pre>
              <pre class="code-block" id="multiCodeSns" style="display: none;"></pre>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- =====================================================================
         SECTION 6: MATPLOTLIB VS SEABORN COMPARISON
         ===================================================================== -->
    <div id="section-mpl-vs-sns" class="section-block">
      <div class="card">
        <div class="card-title">⚖️ Matplotlib vs. Seaborn Classroom Laboratory</div>
        <div class="card-subtitle">
          Compare side-by-side implementation code to clearly understand low-level imperative drawing versus high-level declarative statistical plotting.
        </div>

        <!-- TEACHER NOTE CARDS -->
        <div class="concept-grid">
          <div class="concept-box">
            <h3>🎨 Matplotlib (The Canvas & Brush)</h3>
            <p><strong>Low-level, imperative plotting library.</strong></p>
            <ul>
              <li>Gives total control over every tick, axis, legend, and line.</li>
              <li>Requires 5 to 10 lines of code for routine statistical aggregations.</li>
              <li>Does not understand Pandas DataFrames natively; data must often be extracted into raw lists or NumPy arrays.</li>
            </ul>
          </div>
          <div class="concept-box">
            <h3>📊 Seaborn (The Statistical Visualizer)</h3>
            <p><strong>High-level, declarative library built on top of Matplotlib.</strong></p>
            <ul>
              <li>Built specifically to work seamlessly with Pandas DataFrames.</li>
              <li>Performs statistical computations (means, confidence intervals, KDE) automatically in one function call.</li>
              <li>Easily adds categorical groupings using the <code>hue</code> parameter without manual loops.</li>
            </ul>
          </div>
        </div>

        <!-- CHART SELECTOR FOR SIDE BY SIDE LAB -->
        <div class="controls-panel">
          <div class="control-group">
            <label for="labChartSelect">Select Visualization Task to Compare</label>
            <select id="labChartSelect" onchange="updateLabComparison()">
              <option value="histogram">1. Histogram (Score Distribution with Statistics)</option>
              <option value="bar">2. Bar Chart (Average Score Grouped by Category)</option>
              <option value="box">3. Box Plot (Five-Number Summary across Groups)</option>
              <option value="scatter">4. Scatter Plot (Two Variables with Group Hue)</option>
              <option value="heatmap">5. Correlation Heatmap (Annotated Matrix)</option>
            </select>
          </div>
        </div>

        <!-- SIDE BY SIDE CODE -->
        <div class="lab-grid">
          <div class="lab-col">
            <div class="lab-header mpl">
              <span>Matplotlib Approach (Imperative)</span>
              <span class="tag-badge">plt.*</span>
            </div>
            <div class="lab-body">
              <pre class="code-block" id="labMplCode"></pre>
            </div>
          </div>

          <div class="lab-col">
            <div class="lab-header sns">
              <span>Seaborn Approach (Declarative)</span>
              <span class="tag-badge">sns.*</span>
            </div>
            <div class="lab-body">
              <pre class="code-block" id="labSnsCode"></pre>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- =====================================================================
         SECTION 7: GRAPH COMPARISON GUIDE
         ===================================================================== -->
    <div id="section-comparison-table" class="section-block">
      <div class="card">
        <div class="card-title">📚 Master Graph Comparison Guide</div>
        <div class="card-subtitle">
          Classroom reference table detailing when, why, and how to choose each chart type during exploratory data analysis.
        </div>

        <div class="table-wrapper">
          <table class="ref-table">
            <thead>
              <tr>
                <th>Graph Name</th>
                <th>Purpose</th>
                <th>When to Use</th>
                <th>Example Question</th>
                <th>Data Type</th>
                <th>Limitations</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Bar Chart</strong></td>
                <td>Compare aggregated metrics (e.g., mean) across categories</td>
                <td>When comparing quantitative averages across discrete groups</td>
                <td><em>"What is the average math score by gender?"</em></td>
                <td>Categorical (X) + Numerical (Y)</td>
                <td>Hides internal spread, variance, and outliers within each group</td>
              </tr>
              <tr>
                <td><strong>Line Chart</strong></td>
                <td>Show progression, trend, or continuity across an ordered sequence</td>
                <td>When categories follow an ordered hierarchy or temporal sequence</td>
                <td><em>"Do scores improve as parental education increases?"</em></td>
                <td>Ordered Categorical / Continuous</td>
                <td>Misleading if categories have no inherent order; implies fake continuity</td>
              </tr>
              <tr>
                <td><strong>Histogram</strong></td>
                <td>Examine distribution, skewness, spread, and modal peaks</td>
                <td>When analyzing the frequency distribution of a single numeric feature</td>
                <td><em>"Are student math scores normally distributed or skewed?"</em></td>
                <td>1 Numerical (Continuous)</td>
                <td>Visual appearance is sensitive to the number of bins chosen</td>
              </tr>
              <tr>
                <td><strong>Box Plot</strong></td>
                <td>Display 5-number summary (Min, Q1, Median, Q3, Max) and outliers</td>
                <td>When comparing spreads across groups and detecting extreme anomalies</td>
                <td><em>"Are there outliers in reading scores among students who took test prep?"</em></td>
                <td>1 Numerical (± Categorical)</td>
                <td>Cannot differentiate between unimodal and bimodal distributions</td>
              </tr>
              <tr>
                <td><strong>Scatter Plot</strong></td>
                <td>Investigate correlation, linearity, and clustering between two variables</td>
                <td>When analyzing whether one quantitative variable predicts or relates to another</td>
                <td><em>"Do students who excel in reading also perform well in writing?"</em></td>
                <td>2 Numerical (Continuous)</td>
                <td>Prone to overplotting with large datasets; cannot demonstrate causation</td>
              </tr>
              <tr>
                <td><strong>Heatmap</strong></td>
                <td>Visualize correlation coefficients across all numerical pairs using color</td>
                <td>During initial data exploration to scan for multicollinearity</td>
                <td><em>"Which exam subjects exhibit the strongest correlation?"</em></td>
                <td>Multiple Numerical</td>
                <td>Only captures linear associations; non-linear relations may score near 0</td>
              </tr>
              <tr>
                <td><strong>Count Plot</strong></td>
                <td>Show absolute headcount or frequency of categorical values</td>
                <td>When inspecting class balance or sample sizes across groups</td>
                <td><em>"How many students belong to each race/ethnicity group?"</em></td>
                <td>1 Categorical</td>
                <td>Shows only frequencies; provides no direct insight into test scores</td>
              </tr>
              <tr>
                <td><strong>Pie Chart</strong></td>
                <td>Show relative proportions as parts of a 100% whole</td>
                <td>Only for 2 to 4 distinct categories where shares are intuitive</td>
                <td><em>"What proportion of students receive standard vs free lunch?"</em></td>
                <td>1 Categorical (&le; 4 classes)</td>
                <td>Human visual perception struggles to accurately compare slice angles</td>
              </tr>
              <tr>
                <td><strong>Pair Plot</strong></td>
                <td>Simultaneously plot all bivariate scatter plots and univariate distributions</td>
                <td>At the start of EDA to scan all multi-variable patterns in one view</td>
                <td><em>"How do all 3 subject scores interact when grouped by gender?"</em></td>
                <td>Multiple Numerical (± Hue)</td>
                <td>Computationally expensive; becomes unreadable with many variables</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

  </div>

  <footer>
    <p>Exploratory Data Analysis (EDA) Interactive Classroom Dashboard &bull; Built with Python, Flask, Pandas, Matplotlib &amp; Seaborn</p>
  </footer>

  <!-- =========================================================================
       VANILLA JAVASCRIPT LOGIC
       ========================================================================= -->
  <script>
    let globalSummary = null;

    // --- TAB SWITCHING ---
    function switchSection(sectionId) {
      document.querySelectorAll('.section-block').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.nav-btn').forEach(btn => btn.classList.remove('active'));

      const target = document.getElementById('section-' + sectionId);
      if (target) target.classList.add('active');

      const activeBtn = Array.from(document.querySelectorAll('.nav-btn'))
        .find(btn => btn.getAttribute('onclick').includes(sectionId));
      if (activeBtn) activeBtn.classList.add('active');
    }

    // --- CODE VISIBILITY TOGGLE ---
    function toggleCode(sectionId) {
      const el = document.getElementById(sectionId);
      if (el) el.classList.toggle('visible');
    }

    // --- CODE SUBTAB SWITCHING (MPL vs SNS) ---
    function switchCodeTab(prefix, tab) {
      const mplBtn = event.target.parentElement.children[0];
      const snsBtn = event.target.parentElement.children[1];
      const mplCode = document.getElementById(prefix + 'CodeMpl');
      const snsCode = document.getElementById(prefix + 'CodeSns');

      if (tab === 'mpl') {
        mplBtn.classList.add('active');
        snsBtn.classList.remove('active');
        mplCode.style.display = 'block';
        snsCode.style.display = 'none';
      } else {
        snsBtn.classList.add('active');
        mplBtn.classList.remove('active');
        snsCode.style.display = 'block';
        mplCode.style.display = 'none';
      }
    }

    // --- INITIALIZE & LOAD SUMMARY ---
    async function loadSummary() {
      try {
        const response = await fetch('/api/summary');
        const data = await response.json();

        if (data.status === 'error') {
          document.getElementById('missingFileAlert').style.display = 'flex';
          return;
        }

        globalSummary = data;

        // Fill KPI cards
        document.getElementById('kpiStudents').innerText = data.total_students.toLocaleString();
        document.getElementById('kpiColumns').innerText = data.total_columns;
        document.getElementById('kpiMath').innerText = data.avg_math;
        document.getElementById('kpiReading').innerText = data.avg_reading;
        document.getElementById('kpiWriting').innerText = data.avg_writing;
        document.getElementById('kpiMissing').innerText = data.missing_values;
        document.getElementById('kpiDuplicates').innerText = data.duplicate_count;

        // Build Table Preview
        buildPreviewTable(data.columns, data.preview, data.dtypes);

        // Populate Dropdowns
        populateDropdowns(data.columns, data.numeric_columns, data.categorical_columns);

        // Initialize studios with default plots
        renderUnivariateChart();
        renderBivariateChart();
        renderMultivariateChart();
        updateLabComparison();

      } catch (err) {
        console.error('Failed to load summary:', err);
        document.getElementById('missingFileAlert').style.display = 'flex';
      }
    }

    function buildPreviewTable(columns, rows, dtypes) {
      const thead = document.getElementById('previewTableHead');
      const tbody = document.getElementById('previewTableBody');

      thead.innerHTML = columns.map(col => `
        <th>
          <div>${col}</div>
          <span style="font-size:0.7rem; font-weight:normal; color:#64748b;">${dtypes[col] || ''}</span>
        </th>
      `).join('');

      tbody.innerHTML = rows.map(row => `
        <tr>${columns.map(col => `<td>${row[col]}</td>`).join('')}</tr>
      `).join('');
    }

    function populateDropdowns(allCols, numCols, catCols) {
      const uniSelect = document.getElementById('uniColSelect');
      const biXSelect = document.getElementById('biXSelect');
      const biYSelect = document.getElementById('biYSelect');

      uniSelect.innerHTML = [
        '<optgroup label="Numerical Variables">',
        ...numCols.map(c => `<option value="${c}">${c}</option>`),
        '</optgroup>',
        '<optgroup label="Categorical Variables">',
        ...catCols.map(c => `<option value="${c}">${c}</option>`),
        '</optgroup>'
      ].join('');

      biXSelect.innerHTML = [
        '<optgroup label="Categorical Variables (Groups)">',
        ...catCols.map(c => `<option value="${c}">${c}</option>`),
        '</optgroup>',
        '<optgroup label="Numerical Variables">',
        ...numCols.map(c => `<option value="${c}">${c}</option>`),
        '</optgroup>'
      ].join('');

      biYSelect.innerHTML = [
        '<optgroup label="Numerical Scores">',
        ...numCols.map(c => `<option value="${c}">${c}</option>`),
        '</optgroup>',
        '<optgroup label="Categorical Features">',
        ...catCols.map(c => `<option value="${c}">${c}</option>`),
        '</optgroup>'
      ].join('');

      // Sensible defaults
      if (numCols.includes('math score')) uniSelect.value = 'math score';
      if (catCols.includes('gender')) biXSelect.value = 'gender';
      if (numCols.includes('math score')) biYSelect.value = 'math score';
    }

    // --- DYNAMIC UNIVARIATE CHART TYPE SELECTION ---
    function updateUniChartOptions() {
      const col = document.getElementById('uniColSelect').value;
      const typeSelect = document.getElementById('uniTypeSelect');
      const isNum = globalSummary.numeric_columns.includes(col);

      if (isNum) {
        typeSelect.innerHTML = `
          <option value="histogram">Histogram (Numeric Distribution)</option>
          <option value="box">Box Plot (5-Number Summary)</option>
        `;
      } else {
        typeSelect.innerHTML = `
          <option value="count">Count Plot (Categorical Frequency)</option>
          <option value="pie">Pie Chart (Proportions)</option>
        `;
      }
    }

    // --- RENDER UNIVARIATE CHART ---
    async function renderUnivariateChart() {
      const col = document.getElementById('uniColSelect').value;
      const chartType = document.getElementById('uniTypeSelect').value;

      const img = document.getElementById('uniChartImg');
      const spinner = document.getElementById('uniSpinner');

      img.style.display = 'none';
      spinner.style.display = 'flex';

      try {
        const res = await fetch(`/api/chart?chart_type=${chartType}&x=${encodeURIComponent(col)}`);
        const data = await res.json();

        if (data.success) {
          img.src = data.image;
          img.style.display = 'block';

          // Update Explanation
          document.getElementById('uniTeachTitle').innerText = data.explanation.title;
          document.getElementById('uniTeachPurpose').innerText = data.explanation.purpose;
          document.getElementById('uniTeachWhen').innerText = data.explanation.when_to_use;
          document.getElementById('uniTeachObserve').innerText = data.explanation.what_to_observe;
          document.getElementById('uniTeachLimits').innerText = data.explanation.limitations;

          // Update Code
          document.getElementById('uniCodeMpl').innerText = data.code_matplotlib;
          document.getElementById('uniCodeSns').innerText = data.code_seaborn;
        }
      } catch (err) {
        console.error('Error rendering univariate chart:', err);
      } finally {
        spinner.style.display = 'none';
      }
    }

    // --- RENDER BIVARIATE CHART ---
    async function renderBivariateChart() {
      const xCol = document.getElementById('biXSelect').value;
      const yCol = document.getElementById('biYSelect').value;
      const chartType = document.getElementById('biTypeSelect').value;

      const img = document.getElementById('biChartImg');
      const spinner = document.getElementById('biSpinner');

      img.style.display = 'none';
      spinner.style.display = 'flex';

      try {
        const res = await fetch(`/api/chart?chart_type=${chartType}&x=${encodeURIComponent(xCol)}&y=${encodeURIComponent(yCol)}`);
        const data = await res.json();

        if (data.success) {
          img.src = data.image;
          img.style.display = 'block';

          document.getElementById('biTeachTitle').innerText = data.explanation.title;
          document.getElementById('biTeachPurpose').innerText = data.explanation.purpose;
          document.getElementById('biTeachWhen').innerText = data.explanation.when_to_use;
          document.getElementById('biTeachObserve').innerText = data.explanation.what_to_observe;
          document.getElementById('biTeachLimits').innerText = data.explanation.limitations;

          document.getElementById('biCodeMpl').innerText = data.code_matplotlib;
          document.getElementById('biCodeSns').innerText = data.code_seaborn;
        }
      } catch (err) {
        console.error('Error rendering bivariate chart:', err);
      } finally {
        spinner.style.display = 'none';
      }
    }

    function loadBivariatePreset(x, y, chartType) {
      document.getElementById('biXSelect').value = x;
      document.getElementById('biYSelect').value = y;
      document.getElementById('biTypeSelect').value = chartType;
      renderBivariateChart();
    }

    // --- MULTIVARIATE CONTROLS & RENDER ---
    function updateMultiControls() {
      const type = document.getElementById('multiTypeSelect').value;
      const hueGroup = document.getElementById('multiHueGroup');
      hueGroup.style.display = (type === 'pairplot') ? 'flex' : 'none';
    }

    async function renderMultivariateChart() {
      const chartType = document.getElementById('multiTypeSelect').value;
      const hueVal = document.getElementById('multiHueSelect').value;
      const hue = (hueVal !== 'none') ? hueVal : '';

      const img = document.getElementById('multiChartImg');
      const spinner = document.getElementById('multiSpinner');

      img.style.display = 'none';
      spinner.style.display = 'flex';

      try {
        const res = await fetch(`/api/chart?chart_type=${chartType}&hue=${encodeURIComponent(hue)}`);
        const data = await res.json();

        if (data.success) {
          img.src = data.image;
          img.style.display = 'block';

          document.getElementById('multiTeachTitle').innerText = data.explanation.title;
          document.getElementById('multiTeachPurpose').innerText = data.explanation.purpose;
          document.getElementById('multiTeachWhen').innerText = data.explanation.when_to_use;
          document.getElementById('multiTeachObserve').innerText = data.explanation.what_to_observe;
          document.getElementById('multiTeachLimits').innerText = data.explanation.limitations;

          document.getElementById('multiCodeMpl').innerText = data.code_matplotlib;
          document.getElementById('multiCodeSns').innerText = data.code_seaborn;
        }
      } catch (err) {
        console.error('Error rendering multivariate chart:', err);
      } finally {
        spinner.style.display = 'none';
      }
    }

    // --- MATPLOTLIB VS SEABORN LAB COMPARISON ---
    async function updateLabComparison() {
      const task = document.getElementById('labChartSelect').value;
      try {
        const res = await fetch(`/api/code_comparison?chart=${task}`);
        const data = await res.json();
        document.getElementById('labMplCode').innerText = data.code_matplotlib;
        document.getElementById('labSnsCode').innerText = data.code_seaborn;
      } catch (err) {
        console.error('Error loading lab code:', err);
      }
    }

    // Start on DOM ready
    window.addEventListener('DOMContentLoaded', () => {
      loadSummary();
      updateMultiControls();
    });
  </script>
</body>
</html>
"""


# ==============================================================================
# 19. CREATE FLASK ROUTES
# ==============================================================================
def create_flask_routes(app):
    """
    Registers all web and API routes onto the Flask application instance:
    - '/' : Renders the single-file dashboard HTML
    - '/api/summary' : Returns dataset KPI summary and metadata
    - '/api/chart' : Dynamically generates and returns in-memory chart images with educational details
    - '/api/code_comparison' : Returns side-by-side Matplotlib & Seaborn code for comparison
    """
    @app.route('/')
    def index():
        html_content = create_dashboard_html()
        return render_template_string(html_content)

    @app.route('/api/summary')
    def api_summary():
        df = load_dataset(CSV_FILE_PATH)
        if df is not None:
            df = clean_dataset(df)
        summary = get_dataset_summary(df)
        return jsonify(summary)

    @app.route('/api/chart')
    def api_chart():
        df = load_dataset(CSV_FILE_PATH)
        if df is None:
            return jsonify({"success": False, "error": f"File '{CSV_FILE_PATH}' not found."}), 404

        df = clean_dataset(df)
        chart_type = request.args.get('chart_type', 'histogram')
        x_col = request.args.get('x', 'math score')
        y_col = request.args.get('y', None)
        hue_col = request.args.get('hue', None)
        if hue_col in ['', 'none', 'None']:
            hue_col = None

        try:
            image_b64 = None
            if chart_type == 'histogram':
                image_b64 = create_histogram(df, x_col)
            elif chart_type == 'bar':
                image_b64 = create_bar_chart(df, x_col, y_col)
            elif chart_type == 'line':
                image_b64 = create_line_chart(df, x_col, y_col)
            elif chart_type == 'box':
                image_b64 = create_box_plot(df, y_col if (y_col and y_col in get_numeric_columns(df)) else x_col,
                                            category_column=x_col if (y_col and y_col in get_numeric_columns(df)) else None)
            elif chart_type == 'scatter':
                if not y_col:
                    y_col = get_numeric_columns(df)[1] if len(get_numeric_columns(df)) > 1 else x_col
                image_b64 = create_scatter_plot(df, x_col, y_col, hue_col)
            elif chart_type == 'heatmap':
                image_b64 = create_heatmap(df)
            elif chart_type == 'count':
                image_b64 = create_count_plot(df, x_col)
            elif chart_type == 'pie':
                image_b64 = create_pie_chart(df, x_col)
            elif chart_type == 'pairplot':
                image_b64 = create_pairplot(df, hue=hue_col)
            else:
                return jsonify({"success": False, "error": f"Unknown chart type: {chart_type}"}), 400

            explanation = get_chart_explanation(chart_type)
            code_mpl = generate_matplotlib_code(chart_type, x_column=x_col, y_column=y_col or "reading score", hue_column=hue_col)
            code_sns = generate_seaborn_code(chart_type, x_column=x_col, y_column=y_col or "reading score", hue_column=hue_col)

            return jsonify({
                "success": True,
                "image": image_b64,
                "explanation": explanation,
                "code_matplotlib": code_mpl,
                "code_seaborn": code_sns
            })

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route('/api/code_comparison')
    def api_code_comparison():
        chart_task = request.args.get('chart', 'histogram')
        code_mpl = generate_matplotlib_code(chart_task, x_column="math score", y_column="reading score", hue_column="gender")
        code_sns = generate_seaborn_code(chart_task, x_column="math score", y_column="reading score", hue_column="gender")
        explanation = get_chart_explanation(chart_task)
        return jsonify({
            "success": True,
            "chart": chart_task,
            "code_matplotlib": code_mpl,
            "code_seaborn": code_sns,
            "explanation": explanation
        })


# ==============================================================================
# 20. MAIN APPLICATION ENTRY POINT
# ==============================================================================
def main():
    """
    Main function to configure and start the Flask web application.
    Checks dataset availability and starts the local server.
    """
    app = Flask(__name__)
    create_flask_routes(app)

    print("=" * 70)
    print(" Exploratory Data Analysis (EDA) Interactive Classroom Dashboard")
    print("=" * 70)

    # Validate dataset existence
    df_test = load_dataset(CSV_FILE_PATH)
    if df_test is not None:
        print(f" [+] Dataset successfully detected: '{CSV_FILE_PATH}' ({len(df_test)} rows, {len(df_test.columns)} columns)")
    else:
        print(f" [!] Warning: '{CSV_FILE_PATH}' not found in current directory.")
        print(f"     Please ensure '{CSV_FILE_PATH}' is placed in the same folder.")

    print(" [*] Starting Flask educational dashboard...")
    print(" [*] Access the application in your browser at: http://127.0.0.1:5000")
    print("=" * 70)

    app.run(host='127.0.0.1', port=5000, debug=False)


if __name__ == '__main__':
    main()
