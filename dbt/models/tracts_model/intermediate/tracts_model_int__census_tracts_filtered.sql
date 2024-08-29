select *
from {{ ref('census_tracts_in_city_boundary') }}
where 2010 < year_ and year_ <= 2020
