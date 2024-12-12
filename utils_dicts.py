numerical_features = [
    'Age at Injury','Average Weekly Wage','Birth Year','IME-4 Count',
    'Days_to_First_Hearing','Days_to_C2','Days_to_C3',
    'Number of Dependents',
    'C-2 Date_Year',
    'C-3 Date_Year',
    'C-2 Date_Month',
    'C-2 Date_Day',
    'C-2 Date_DayOfWeek',
    'C-3 Date_Month',
    'C-3 Date_Day',
    'C-3 Date_DayOfWeek',
    'First Hearing Date_Month',
    'First Hearing Date_Day',
    'First Hearing Date_DayOfWeek',
    'Days_to_C2',
    'Days_to_C3',
    'First Hearing Date_Year','First Hearing Date_Month','First Hearing Date_Day','First Hearing Date_DayOfWeek'
]
categorical_features = [
    'Alternative Dispute Resolution',
    'Enc County of Injury', 'Enc District Name','Enc Industry Code',
    'Medical Fee Region_II','Medical Fee Region_III','Medical Fee Region_IV',
    'Known Accident Date','Known Assembly Date','Known C-2 Date','Known C-3 Date',
    'Enc WCIO Cause of Injury Code',
    'Enc WCIO Nature of Injury Code',
    'Enc WCIO Part Of Body Code',
    'Enc Zip Code',
    'Attorney/Representative_Y',
    'Carrier Type_2A. SIF',
    'Carrier Type_3A. SELF PUBLIC',
    'Carrier Type_4A. SELF PRIVATE',
    'Carrier Type_5D. SPECIAL FUND - UNKNOWN',
    'Carrier Type_UNKNOWN',
    'COVID-19 Indicator_Y',
    'Gender_Unknown',
    'Medical Fee Region_UK',
    'Known First Hearing Date','Known Age at Injury','Known Birth Year','Accident Date_Year',
    'Accident Date_Month','Accident Date_Day','Accident Date_DayOfWeek','Assembly Date_Year',
    'Assembly Date_Month','Assembly Date_Day','Assembly Date_DayOfWeek',
    'Holiday_Accident','Weekend_Accident', 'Risk_Level','Gender_M',
    'Accident_Season_Sin','Accident_Season_Cos'
]

"""
essential_features = [
    "IME-4 Count",
    "WCIO Nature of Injury Code",
    "Years Past Accident",
    "Industry Code",
    "Average Weekly Wage",
    # Additional categorical features
    "Carrier Name",
    "Carrier Type",
    "County of Injury",
    "District Name",
    "Gender",
    "Medical Fee Region",
    "Attorney/Representative",
    "COVID-19 Indicator",
    "First Hearing Date Occurred",
    "C-2 Date Occurred",
    "C-3 Date Occurred",
    "Birth Year Occurred",
    "Age at Injury Occurred",
    "Accident Date Occurred"
]

reduced_features = [
    "Age at Injury",
    "IME-4 Count",
    "Days_to_First_Hearing",
    "Average Weekly Wage",
    "Birth Year",
    "C-2 Date_Year",
    "C-3 Date_Year",
    "First Hearing Date_Year",
    "First Hearing Date_Month"
    # Additional categorical features
    "County of Injury",
    "District Name",
    "Industry Code",
    "Medical Fee Region",
    "Attorney/Representative",
    "COVID-19 Indicator",
    "Known C-2 Date",
    "Known C-3 Date",
    "Known First Hearing Date",
    "Accident Date_Year",
    "Accident Date_Month",
    "Accident Date_Day",
    "Gender_F",
    "Gender_M",
    "Weekend_Accident"
]


all_features = numerical_features + categorical_features
"""
