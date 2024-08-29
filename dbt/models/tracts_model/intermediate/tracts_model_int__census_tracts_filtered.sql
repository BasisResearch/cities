select *
from {{ ref('census_tracts_in_city_boundary') }}
where year_ <= 2020
