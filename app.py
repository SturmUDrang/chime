from functools import reduce
from typing import Tuple, Dict, Any
import pandas as pd
import streamlit as st
import numpy as np
import altair as alt

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

delaware = 564696
chester = 519293
montgomery = 826075
bucks = 628341
philly = 1581000
S_default = 9700000
#delaware + chester + montgomery + bucks + philly
known_infections = 73 # update daily
cases_percent = 25
known_cases = int(known_infections * (  cases_percent / 100 ))

a = """Mi változott az eredeti modellhez képest?
 - Esetszám duplázodás az eredeti modell szerint 6-ról fel lett emelve 7-re ami jobban közelít az általam olvasott statisztikákhoz.
 - Az amerikai adatokkal ellentétben a modell a teljes ország egészségügyi rendszerének terhelését mutatja, nem csak egy kórházét.
 - A lakosság száma módosítva lett

"""


# Widgets
current_hosp = st.sidebar.number_input(
    "Jelenleg valóban kórházban ápolásra szoruló ferőzöttek", value=known_cases, step=1, format="%i"
)
st.sidebar.markdown("""Becsült érték, az ismert {known_infections} eset {cases_percent}%-a. Amennyiben pontos adat rendelkezésre áll a fenti érték módosítandó""".format(known_infections=known_infections, cases_percent=cases_percent))

doubling_time = st.sidebar.number_input(
    "Esetszám duplázódás (napok)", value=7, step=1, format="%i"
)

relative_contact_rate = st.sidebar.number_input(
    "Társadalmi érintekzések csökkenése (%)", 0, 100, value=0, step=5, format="%i"
)/100.0

hosp_rate = (
    st.sidebar.number_input("Kórházi ellátást igényelő beteg (az összes fertőzés %-ban)", 0.0, 100.0, value=5.0, step=1.0, format="%f")
    / 100.0
)
icu_rate = (
    st.sidebar.number_input("Intenzív osztályon kezelt (az összes fertőzés %-ban)", 0.0, 100.0, value=2.0, step=1.0, format="%f") / 100.0
)
vent_rate = (
    st.sidebar.number_input("Gép által lélelgeztetett (az összes fertőzés %-ban)", 0.0, 100.0, value=1.0, step=1.0, format="%f")
    / 100.0
)
hosp_los = st.sidebar.number_input("Kórházi kezelés hossza (nap)", value=7, step=1, format="%i")
icu_los = st.sidebar.number_input("Intenzív kezelés hossza (nap)", value=9, step=1, format="%i")
vent_los = st.sidebar.number_input("Lélegeztetés hossza (nap)", value=10, step=1, format="%i")
Penn_market_share = 1
# (
#    st.sidebar.number_input(
#        "Ágyak részesedése (%)", 0.0, 100.0, value=15.0, step=1.0, format="%f"
#    )
#    / 100.0
#)
S = st.sidebar.number_input(
    "Lakosság", value=S_default, step=100000, format="%i"
)

initial_infections = st.sidebar.number_input(
    "Jelenleg ismert fertőzöttek (nem módosít az előrejelzéseken, csak a felfedezési arány számolásához szükséges)", value=known_infections, step=10, format="%i"
)

total_infections = current_hosp / Penn_market_share / hosp_rate
detection_prob = initial_infections / total_infections

S, I, R = S, initial_infections / detection_prob, 0

intrinsic_growth_rate = 2 ** (1 / doubling_time) - 1

recovery_days = 14.0
# mean recovery rate, gamma, (in 1/days).
gamma = 1 / recovery_days

# Contact rate, beta
beta = (
    intrinsic_growth_rate + gamma
) / S * (1-relative_contact_rate) # {rate based on doubling time} / {initial S}

r_t = beta / gamma * S # r_t is r_0 after distancing
r_naught = r_t / (1-relative_contact_rate)
doubling_time_t = 1/np.log2(beta*S - gamma +1) # doubling time after distancing

def head():
    st.title("Koronavírusfertőzöttek ellátásához szükséges kórházi ágyak száma Magyarországon")
    st.markdown(
        """A szoftver és a modell eredeti verzióját a [Predictive Healthcare team](http://predictivehealthcare.pennmedicine.org/) a Pennsylvaniai Egyetem orvosi központjában fejlesztette. 
Az eredeti fejlesztőcsapat elérhetősge: (http://predictivehealthcare.pennmedicine.org/contact/). 
A magyar valtozat forráskódja a [GitHub](https://github.com/SturmUDrang/chime)-on található.

A magyar változatott fordította és a modell sarokszámait testreszabta Lőrincze Tamás.""")


    st.markdown(
        """A modell által becsült valós fertőzések száma **{total_infections:.0f}**. A(z) **{initial_infections}**
    hivatalosan jelentett fertőzés **{detection_prob:.0%}** észlelési arányt mutat. Az érték a jelenleg kórházban ápoltak száma (**{current_hosp}**), a kózházi ápolást igénylő betegek százalékának (**{hosp_rate:.0%}**) és a lakosság számának (**{S}**) függvénye, becslés.

Az esetszám duplázódás sebességéből következik, hogy minden fertőzött átlagban  $R_0$ =
**{r_naught:.2f}** további embernek adja át a betegséget. 

*Társadalmi érintekzések csökkenése* (amennyiben be lett állítva): **{relative_contact_rate:.0%}**-os csökkenés a társadalmi érintkezésekben az esetszám duplázódást **{doubling_time_t:.1f}** napra növeli, azaz az egy fertőzött által továbbfertőzőtt egyének száma: $R_t$ = **${r_t:.2f}** 
""".format(
        total_infections=total_infections,
        initial_infections=initial_infections,
        detection_prob=detection_prob,
        current_hosp=current_hosp,
        hosp_rate=hosp_rate,
        S=S,
        Penn_market_share=Penn_market_share,
        recovery_days=recovery_days,
        r_naught=r_naught,
        doubling_time=doubling_time,
        relative_contact_rate=relative_contact_rate,
        r_t=r_t,
        doubling_time_t=doubling_time_t
    )
    )

    return None

head()



st.markdown("""A modellról részletes információt az [eredeti oldal](http://penn-chime.phl.io/) tartalmaz""")

st.markdown("""A bal oldali menüt kinyitva lehet a feltételezéseken változtatni, pl. a *Társadalmi érintekzések csökkenése (%)* paraméter megváltoztatásával módosul a járvány terjedési sebessége""")

st.markdown(a)

# The SIR model, one time step
def sir(y, beta, gamma, N):
    S, I, R = y
    Sn = (-beta * S * I) + S
    In = (beta * S * I - gamma * I) + I
    Rn = gamma * I + R
    if Sn < 0:
        Sn = 0
    if In < 0:
        In = 0
    if Rn < 0:
        Rn = 0

    scale = N / (Sn + In + Rn)
    return Sn * scale, In * scale, Rn * scale


# Run the SIR model forward in time
def sim_sir(S, I, R, beta, gamma, n_days, beta_decay=None):
    N = S + I + R
    s, i, r = [S], [I], [R]
    for day in range(n_days):
        y = S, I, R
        S, I, R = sir(y, beta, gamma, N)
        if beta_decay:
            beta = beta * (1 - beta_decay)
        s.append(S)
        i.append(I)
        r.append(R)

    s, i, r = np.array(s), np.array(i), np.array(r)
    return s, i, r


n_days = st.slider("Hány napot modellezzen", 30, 200, 90, 1, "%i")

beta_decay = 0.0
s, i, r = sim_sir(S, I, R, beta, gamma, n_days, beta_decay=beta_decay)


hosp = i * hosp_rate * Penn_market_share
icu = i * icu_rate * Penn_market_share
vent = i * vent_rate * Penn_market_share

days = np.array(range(0, n_days + 1))
data_list = [days, hosp, icu, vent]
data_dict = dict(zip(["day", "hosp", "icu", "vent"], data_list))

projection = pd.DataFrame.from_dict(data_dict)

st.subheader("Kózházi kezelésre felvettek")
st.markdown("Az újonnan kórházi kezelésre szoroló betegek száma **naponta**")

# New cases
projection_admits = projection.iloc[:-1, :] - projection.shift(1)
projection_admits[projection_admits < 0] = 0

plot_projection_days = n_days - 10
projection_admits["day"] = range(projection_admits.shape[0])


def new_admissions_chart(projection_admits: pd.DataFrame, plot_projection_days: int) -> alt.Chart:
    """docstring"""
    projection_admits = projection_admits.rename(columns={"hosp": "Kórházban kezelt", "icu": "Intenzív osztályon kezelt", "vent": "Lélegeztetett"})
    return (
        alt
        .Chart(projection_admits.head(plot_projection_days))
        .transform_fold(fold=["Kórházban kezelt", "Intenzív osztályon kezelt", "Lélegeztetett"])
        .mark_line(point=True)
        .encode(
            x=alt.X("day", title="Mostantól nap"),
            y=alt.Y("value:Q", title="Naponta kórházi kezelésre szoroló új estetek"),
            color="key:N",
            tooltip=["day", "key:N"]
        )
        .interactive()
    )

st.altair_chart(new_admissions_chart(projection_admits, plot_projection_days), use_container_width=True)



#if st.checkbox("Show Projected Admissions in tabular form"):
#    admits_table = projection_admits[np.mod(projection_admits.index, 7) == 0].copy()
#    admits_table["day"] = admits_table.index
#    admits_table.index = range(admits_table.shape[0])
#    admits_table = admits_table.fillna(0).astype(int)

#    st.dataframe(admits_table)

st.subheader("Kórházi kezelésre szoruló betegek **összesen**")
st.markdown(
    "Az összes kórházi ápolásra szoroló beteg száma, azaz a szükséges ágyak, intenzív helyek és lélegeztetőgépek száma"
)

def _census_table(projection_admits, hosp_los, icu_los, vent_los) -> pd.DataFrame:
    """ALOS for each category of COVID-19 case (total guesses)"""

    los_dict = {
        "hosp": hosp_los,
        "icu": icu_los,
        "vent": vent_los,
    }

    census_dict = dict()
    for k, los in los_dict.items():
        census = (
            projection_admits.cumsum().iloc[:-los, :]
            - projection_admits.cumsum().shift(los).fillna(0)
        ).apply(np.ceil)
        census_dict[k] = census[k]


    census_df = pd.DataFrame(census_dict)
    census_df["day"] = census_df.index
    census_df = census_df[["day", "hosp", "icu", "vent"]]

    census_table = census_df[np.mod(census_df.index, 7) == 0].copy()
    census_table.index = range(census_table.shape[0])
    census_table.loc[0, :] = 0
    census_table = census_table.dropna().astype(int)

    return census_table

census_table = _census_table(projection_admits, hosp_los, icu_los, vent_los)

def admitted_patients_chart(census: pd.DataFrame) -> alt.Chart:
    """docstring"""
    census = census.rename(columns={"hosp": "Kórházban kezelt", "icu": "Intenzív osztályon kezelt", "vent": "Lélegeztetett"})

    return (
        alt
        .Chart(census)
        .transform_fold(fold=["Kórházban kezelt", "Intenzív osztályon kezelt", "Lélegeztetett"])
        .mark_line(point=True)
        .encode(
            x=alt.X("day", title="Mostantól nap"),
            y=alt.Y("value:Q", title="Össz"),
            color="key:N",
            tooltip=["day", "key:N"]
        )
        .interactive()
    )

st.altair_chart(admitted_patients_chart(census_table), use_container_width=True)

#if st.checkbox("Show Projected Census in tabular form"):
#    st.dataframe(census_table)

def additional_projections_chart(i: np.ndarray, r: np.ndarray) -> alt.Chart:
    dat = pd.DataFrame({"Infected": i, "Recovered": r})

    return (
        alt
        .Chart(dat.reset_index())
        .transform_fold(fold=["Infected", "Recovered"])
        .mark_line()
        .encode(
            x=alt.X("index", title="Days from today"),
            y=alt.Y("value:Q", title="Case Volume"),
            tooltip=["key:N", "value:Q"],
            color="key:N"
        )
        .interactive()
    )

#st.markdown(
#    """**Click the checkbox below to view additional data generated by this simulation**"""
#)

def show_additional_projections():
    st.subheader(
        "The number of infected and recovered individuals in the hospital catchment region at any given moment"
    )

    st.altair_chart(additional_projections_chart(i, r), use_container_width=True)

    if st.checkbox("Show Raw SIR Similation Data"):
        # Show data
        days = np.array(range(0, n_days + 1))
        data_list = [days, s, i, r]
        data_dict = dict(zip(["day", "susceptible", "infections", "recovered"], data_list))
        projection_area = pd.DataFrame.from_dict(data_dict)
        infect_table = (projection_area.iloc[::7, :]).apply(np.floor)
        infect_table.index = range(infect_table.shape[0])

        st.dataframe(infect_table)


#if st.checkbox("Show Additional Projections"):
#    show_additional_projections()


# Definitions and footer

st.subheader("Guidance on Selecting Inputs")
st.markdown(
    """* **Hospitalized COVID-19 Patients:** The number of patients currently hospitalized with COVID-19 **at your hospital(s)**. This number is used in conjunction with Hospital Market Share and Hospitalization % to estimate the total number of infected individuals in your region.
* **Doubling Time (days):** This parameter drives the rate of new cases during the early phases of the outbreak. The American Hospital Association currently projects doubling rates between 7 and 10 days. This is the doubling time you expect under status quo conditions. To account for reduced contact and other public health interventions, modify the _Social distancing_ input.
* **Social distancing (% reduction in person-to-person physical contact):** This parameter allows users to explore how reduction in interpersonal contact & transmission (hand-washing) might slow the rate of new infections. It is your estimate of how much social contact reduction is being achieved in your region relative to the status quo. While it is unclear how much any given policy might affect social contact (eg. school closures or remote work), this parameter lets you see how projections change with percentage reductions in social contact.
* **Hospitalization %(total infections):** Percentage of **all** infected cases which will need hospitalization.
* **ICU %(total infections):** Percentage of **all** infected cases which will need to be treated in an ICU.
* **Ventilated %(total infections):** Percentage of **all** infected cases which will need mechanical ventilation.
* **Hospital Length of Stay:** Average number of days of treatment needed for hospitalized COVID-19 patients.
* **ICU Length of Stay:** Average number of days of ICU treatment needed for ICU COVID-19 patients.
* **Vent Length of Stay:**  Average number of days of ventilation needed for ventilated COVID-19 patients.
* **Hospital Market Share (%):** The proportion of patients in the region that are likely to come to your hospital (as opposed to other hospitals in the region) when they get sick. One way to estimate this is to look at all of the hospitals in your region and add up all of the beds. The number of beds at your hospital divided by the total number of beds in the region times 100 will give you a reasonable starting estimate.
* **Regional Population:** Total population size of the catchment region of your hospital(s).
* **Currently Known Regional Infections**: The number of infections reported in your hospital's catchment region. This is only used to compute detection rate - **it will not change projections**. This input is used to estimate the detection rate of infected individuals.
    """
)


st.subheader("References & Acknowledgements")
st.markdown(
    """* AHA Webinar, Feb 26, James Lawler, MD, an associate professor University of Nebraska Medical Center, What Healthcare Leaders Need To Know: Preparing for the COVID-19
* We would like to recognize the valuable assistance in consultation and review of model assumptions by Michael Z. Levy, PhD, Associate Professor of Epidemiology, Department of Biostatistics, Epidemiology and Informatics at the Perelman School of Medicine
    """
)
st.markdown("© 2020, The Trustees of the University of Pennsylvania")
