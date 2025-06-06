# 🧪 Single- and Multi-Market Simulation Based Forecast for Drug X

This repository provides a modular and vectorized simulation framework for forecasting patient uptake of a new drug (Drug X) across multiple countries. The goal is to support strategic planning with an uncertainty-aware model of market dynamics and treatment persistence.

---

## 📚 Notebooks Overview

| Notebook | Description |
|----------|-------------|
| **01_Forecast_Patient_Uptake_for_DrugX** | Builds a basic stichastic forecast for a single market. |
| **02_Simulation_Based_Forecast** | Extends the logic to generate a funnel of possible forecasts for a single market by running several simulations. |
| **03_Simulation_Based_Forecast_Multi_Markets** | Runs simulation-based forecasts for Drug X adoption and persistence across multiple countries.. |
| **04_Vectorized_Forecast_Patient_Uptake_for_DrugX** | Refactors simulation logic using NumPy vectorization to support faster execution and scalability across simulations. |
| **05_Vectorized_Simulation_Based_Forecast_Multi_Markets** | Implements a vectorized simulation framework to model patient uptake of Drug X across multiple countries and simulation runs. |

---

## 🧬 Simulation Logic Overview
Below is a diagram showing how the simulation steps are chained:
![Simulation Diagram](images/DGP_Diagram.jpg)

All simulation functions are reusable, modular, well-documented, and stored in `utils.py` and `utils_vect.py`

---

## 🎯 Applications

This framework enables:
- **Market sizing and prioritization**
- **Scenario analysis under uncertainty**

With a solid patient-level forecast, future extensions could integrate cost structures, pricing scenarios, or logistics planning.

---

## 🤖 Technology Used
- **Python**
- **Jupyter Notebooks**
- `pandas`, `numpy`
- `matplotlib.pyplot`
- **LLMs Used:** ChatGPT
----
## 📬 Contact

For feedback, suggestions, or collaboration:

- **Carlo Signorini**
- [LinkedIn Profile](https://www.linkedin.com/in/carlosignorini/)

---

## 🙏 Acknowledgments

This project benefited from the use of AI assistants such as **ChatGPT**, which was used extensively for designing simulation logic, vectorization strategies, visualization techniques, documentation structure, and some explanation texts.

---

## 📄 License

This project is licensed under the **MIT License**.

> You are free to use, modify, and distribute this project for personal or commercial purposes. Attribution is appreciated but not required.

---

--- 
## 📊 Example Output
Here are some visual outputs generated by the simulation framework.

**5 simulations of persistence on Drug X over time:**

![Persistence on Drug X](images/persistence.png)

**Median Patient Forecast with Uncertainty Bands per Country:**
this plot shows forecasted monthly patients on treatment by country, including the 10th–90th percentile uncertainty band.

![Persistence on Drug X](images/multimarket_forecast.png)


