import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Creating containers
header = st.container()
data_sets = st.container()
visualization = st.container()
model_training = st.container()
performance_metrics = st.container()



with header:
    st.title("Titanic Dataset App")
    st.text("This app uses RandomForestClassifier to predict survival on the Titanic.")
    st.subheader("Some features of the app are: ")
    st.markdown("1. Titanic dataset overview")
    st.markdown("2. Data Vizualization: Matplotlib and Pyplot")
    st.markdown("3. Model Training and Outcomes")
    st.markdown("4. Prediction wrt user input")



with data_sets:
    st.header("Titanic Dataset")
    st.text("The following data has been label encoded also applying standardization: ")
    # Importing data
    df = sns.load_dataset('titanic')
    df.ffill(inplace=True)

# ///   PREPROCESSING
    #LABEL ENCODER
    label_encoder = LabelEncoder()
    df["class"] = label_encoder.fit_transform(df["class"])
    df["sex"] = label_encoder.fit_transform(df["sex"])
    df["embark_town"] = label_encoder.fit_transform(df["embark_town"])
    df["embarked"] = label_encoder.fit_transform(df["embarked"])
    df["alive"] = label_encoder.fit_transform(df["alive"])
    df["alone"] = label_encoder.fit_transform(df["alone"])

    #STANDARDIZATION
    features = ["pclass", "sex", "age", "parch", "fare", "embarked"]
    x = df[features]
    y = df["survived"]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Splitting the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42)
#///

    st.write(df.head(10))



with visualization:
    st.header("Data Visualization")

    st.subheader("Male vs Female Distribution")
    gender_counts = df["sex"].value_counts()
    st.bar_chart(gender_counts)  


    st.subheader("Class Distribution")
    class_counts = df["class"].value_counts()
    st.area_chart(class_counts)


    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader("Age Distribution")
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x="age", kde=True)
    st.pyplot()


    st.subheader("Age vs Passenger Class")
    fig, ax = plt.subplots()
    for x in [1, 2, 3]:
        sns.kdeplot(data=df[df["class"] == x], x="age", ax=ax, label=f"{x}st")
    plt.title("Age vs Passenger Class")
    plt.legend()
    st.pyplot(fig)


    st.subheader("FacetGrid: Survival by Gender, Age, and Fare")
    g = sns.FacetGrid(df, hue="survived", col="sex", margin_titles=True,
                      palette="Set1", hue_kws=dict(marker=["^", "v"]))
    g.map(plt.scatter, "fare", "age", edgecolor="w")
    g.add_legend()

    plt.subplots_adjust(top=0.8)
    g.fig.suptitle('Survival by Gender, Age, and Fare')
    st.pyplot()


    st.subheader("Categorical Plot: Survival by Deck")
    g = sns.catplot(x="survived", col="deck", col_wrap=4,
                    data=df[df["deck"].notnull()],
                    kind="count", height=2.5, aspect=0.8)
    
    g.fig.subplots_adjust(top=1)
    g.fig.suptitle("Survival by Deck")
    st.pyplot()


    st.subheader("Age Distribution")
    sns.set_context("notebook", font_scale=1.0)
    sns.set_style("whitegrid")

    fig = sns.displot(df["age"].dropna(),
                      bins=80,
                      kde=False,
                      color="red")
    plt.title("Age Distribution")
    plt.ylabel("Count")
    st.pyplot(fig)




with model_training:    
    st.header("Model Training")
    # Random Forest Classifier parameters
    n_estimators = st.slider("Number of trees", min_value=50, max_value=300, step=50, value=100)
    max_depth = st.slider("Maximum tree depth", min_value=5,
                          max_value=20, step=5, value=10)

    # Create and train the Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)


with performance_metrics:
    
    #performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)  -> not for classifiers
    roc_auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.header("The performance metrics of the model are: ")
    st.write("**Model accuracy:**", accuracy)
    st.write("**Model precision:** ", precision)
    st.write("**Model roc-auc score:**", roc_auc)
    st.write("**Confusion Matrix:** ")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
    class_labels = ["Positive", "Negative"] #class labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# Replace tick labels with class labels
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    plt.title("Confusion Matrix")
    st.pyplot(fig)



st.write("**The Random Forest Classifier has been succesfully implemented.\nThe perfomance metrics prove that it has achieved substantial results.**")
