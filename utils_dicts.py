numerical_features = [
    'Age at Injury','Average Weekly Wage','Birth Year','IME-4 Count',
    'Number of Dependents','Days_to_First_Hearing','Days_to_C2','Days_to_C3',
    'C-2 Date_Year','C-2 Date_Month','C-2 Date_Day','C-2 Date_DayOfWeek',
    'C-3 Date_Year','C-3 Date_Month','C-3 Date_Day', 'C-3 Date_DayOfWeek',
    'First Hearing Date_Year','First Hearing Date_Month','First Hearing Date_Day','First Hearing Date_DayOfWeek'
]
categorical_features = [
    'Enc County of Injury', 'Enc District Name','Enc Industry Code',
    'Medical Fee Region_I','Medical Fee Region_II','Medical Fee Region_III','Medical Fee Region_IV',
    'Known Accident Date','Known Assembly Date','Known C-2 Date','Known C-3 Date',
    'Enc WCIO Cause of Injury Code',
    'Nature_Injury_Code_Hash_0','Nature_Injury_Code_Hash_1','Nature_Injury_Code_Hash_2',
    'Nature_Injury_Code_Hash_3','Nature_Injury_Code_Hash_4','Nature_Injury_Code_Hash_5',
    'Nature_Injury_Code_Hash_6','Nature_Injury_Code_Hash_7','Nature_Injury_Code_Hash_8',
    'Nature_Injury_Code_Hash_9','Nature_Injury_Code_Hash_10','Nature_Injury_Code_Hash_11',
    'Nature_Injury_Code_Hash_12','Nature_Injury_Code_Hash_13','Nature_Injury_Code_Hash_14',
    'Nature_Injury_Code_Hash_15','Nature_Injury_Code_Hash_16','Nature_Injury_Code_Hash_17',
    'Nature_Injury_Code_Hash_18','Nature_Injury_Code_Hash_19',
    'Part_Of_Body_Code_Hash_0','Part_Of_Body_Code_Hash_1','Part_Of_Body_Code_Hash_2',
    'Part_Of_Body_Code_Hash_3','Part_Of_Body_Code_Hash_4','Part_Of_Body_Code_Hash_5',
    'Part_Of_Body_Code_Hash_6','Part_Of_Body_Code_Hash_7','Part_Of_Body_Code_Hash_8',
    'Part_Of_Body_Code_Hash_9','Part_Of_Body_Code_Hash_10','Part_Of_Body_Code_Hash_11',
    'Part_Of_Body_Code_Hash_12','Part_Of_Body_Code_Hash_13','Part_Of_Body_Code_Hash_14',
    'Part_Of_Body_Code_Hash_15','Part_Of_Body_Code_Hash_16','Part_Of_Body_Code_Hash_17',
    'Part_Of_Body_Code_Hash_18','Part_Of_Body_Code_Hash_19',
    'Zip_Code_Hash_0','Zip_Code_Hash_1',
    'Zip_Code_Hash_2','Zip_Code_Hash_3','Zip_Code_Hash_4',
    'Zip_Code_Hash_5','Zip_Code_Hash_6','Zip_Code_Hash_7',
    'Zip_Code_Hash_8','Zip_Code_Hash_9','Zip_Code_Hash_10',
    'Zip_Code_Hash_11','Zip_Code_Hash_12','Zip_Code_Hash_13',
    'Zip_Code_Hash_14','Zip_Code_Hash_15','Zip_Code_Hash_16',
    'Zip_Code_Hash_17','Zip_Code_Hash_18','Zip_Code_Hash_19',
    'Known First Hearing Date','Known Age at Injury','Known Birth Year','Accident Date_Year',
    'Accident Date_Month','Accident Date_Day','Accident Date_DayOfWeek','Assembly Date_Year',
    'Assembly Date_Month','Assembly Date_Day','Assembly Date_DayOfWeek',
    'Holiday_Accident','Weekend_Accident', 'Risk_Level','Gender_F','Gender_M',
    'Accident_Season_Sin','Accident_Season_Cos'
]


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

reduced_features = numerical_features = [
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
