with
census_tracts as (select * from {{ ref('tracts_model_int__census_tracts_filtered') }})
select parcels.*
from {{ ref('parcels') }} join census_tracts using (census_tract_id)
