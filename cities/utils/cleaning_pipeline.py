from cities.utils.cleaning_scripts.clean_age_composition import clean_age_composition
from cities.utils.cleaning_scripts.clean_burdens import clean_burdens
from cities.utils.cleaning_scripts.clean_ethnic_composition import (
    clean_ethnic_composition,
)
from cities.utils.cleaning_scripts.clean_ethnic_composition_ma import (
    clean_ethnic_composition_ma,
)
from cities.utils.cleaning_scripts.clean_gdp import clean_gdp
from cities.utils.cleaning_scripts.clean_gdp_ma import clean_gdp_ma
from cities.utils.cleaning_scripts.clean_hazard import clean_hazard
from cities.utils.cleaning_scripts.clean_homeownership import clean_homeownership
from cities.utils.cleaning_scripts.clean_income_CT import clean_income_CT
from cities.utils.cleaning_scripts.clean_income_distribution import (
    clean_income_distribution,
)
from cities.utils.cleaning_scripts.clean_industry import clean_industry
from cities.utils.cleaning_scripts.clean_industry_ma import clean_industry_ma
from cities.utils.cleaning_scripts.clean_industry_ts import clean_industry_ts
from cities.utils.cleaning_scripts.clean_population import clean_population
from cities.utils.cleaning_scripts.clean_population_CT import clean_population_CT
from cities.utils.cleaning_scripts.clean_population_density import (
    clean_population_density,
)
from cities.utils.cleaning_scripts.clean_population_ma import clean_population_ma
from cities.utils.cleaning_scripts.clean_spending_commerce import (
    clean_spending_commerce,
)
from cities.utils.cleaning_scripts.clean_spending_HHS import clean_spending_HHS
from cities.utils.cleaning_scripts.clean_spending_transportation import (
    clean_spending_transportation,
)
from cities.utils.cleaning_scripts.clean_transport import clean_transport
from cities.utils.cleaning_scripts.clean_unemployment import clean_unemployment
from cities.utils.cleaning_scripts.clean_urbanicity_ct import clean_urbanicity_CT
from cities.utils.cleaning_scripts.clean_urbanicity_ma import clean_urbanicity_ma
from cities.utils.cleaning_scripts.clean_urbanization import clean_urbanization
from cities.utils.cleaning_scripts.cleaning_poverty import clean_poverty

# from cities.utils.clean_health import clean_health


# clean_health() lost of another 15-ish fips
clean_income_CT()

clean_urbanicity_CT()

clean_population_CT()

clean_population_density()

clean_homeownership()

clean_income_distribution()

clean_hazard()

clean_burdens()

clean_age_composition()

clean_gdp_ma()

clean_industry_ma()

clean_urbanicity_ma()

clean_ethnic_composition_ma()

clean_population_ma()

clean_poverty()

clean_unemployment()

clean_gdp()

clean_population()

clean_transport()

clean_spending_transportation()

clean_spending_commerce()

clean_spending_HHS()

clean_ethnic_composition()

clean_industry()

clean_urbanization()

clean_industry_ts()
