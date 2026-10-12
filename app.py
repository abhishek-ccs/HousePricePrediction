
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

# Load your existing regression model
regression = joblib.load("house_model.pkl")

st.title("🏡 House Price Prediction")
st.write("Enter your house details below.")

area = st.number_input("Area (sq. ft.)", min_value=100, value=1200)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, value=2)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, value=2)
parking = st.number_input("Parking Spaces", min_value=0, value=1)
city = st.text_input("Enter City", value="Delhi").strip()
floors = st.number_input("Number Of Floors",min_value = 1, value = 1)

# Get city categories from the trained model
city_columns = [
    col for col in regression.feature_names_in_
    if col.startswith("City_")
]

known_cities = {
    col.replace("City_", "").casefold(): col
    for col in city_columns
}

if st.button("Predict Price"):

    if not city:
        st.error("Please enter a city.")
        st.stop()

    # Numeric house features
    base_data = {
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Parking": parking,
        "Floors":floors
    }

    normalized_city = city.casefold()

    if normalized_city in known_cities:
        # Known city: use its actual trained city feature
        input_data = pd.DataFrame([{
            **base_data,
            **{
                col: int(col == known_cities[normalized_city])
                for col in city_columns
            }
        }])

        prediction = regression.predict(input_data)[0]

        st.success(f"Predicted Price: ₹{prediction:,.0f}")
        st.caption("Prediction uses the selected city's trained feature.")

    else:
        # Unknown city: estimate using the average predictions
        # across all cities known to the model.
        fallback_rows = []

        for city_feature in city_columns:
            row = {
                **base_data,
                **{col: 0 for col in city_columns}
            }
            row[city_feature] = 1
            fallback_rows.append(row)

        fallback_data = pd.DataFrame(fallback_rows)
        fallback_data = fallback_data.reindex(
            columns=regression.feature_names_in_
        )

        city_predictions = regression.predict(fallback_data)
        prediction = city_predictions.mean()

        st.success(f"Estimated Price: ₹{prediction:,.0f}")
        st.info(
            f"{city} was not in the training dataset. "
            "This is an average-city estimate, not a "
            "city-specific prediction."
        )
