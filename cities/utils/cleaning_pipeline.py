from cities.utils.clean_ethnic_composition import clean_ethnic_composition
from cities.utils.clean_gdp import clean_gdp
from cities.utils.clean_industry import clean_industry
from cities.utils.clean_industry_ts import clean_industry_ts
from cities.utils.clean_population import clean_population
from cities.utils.clean_spending_commerce import clean_spending_commerce
from cities.utils.clean_spending_HHS import clean_spending_HHS
from cities.utils.clean_spending_transportation import clean_spending_transportation
from cities.utils.clean_transport import clean_transport
from cities.utils.clean_unemployment import clean_unemployment
from cities.utils.clean_urbanization import clean_urbanization
from cities.utils.cleaning_poverty import clean_poverty 

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

clean_poverty()
