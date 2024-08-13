with
parcels_to_zip_codes as (
  select
    parcel_id
    , zip_code_id
  from {{ref('parcels_to_zip_codes')}}
),
parcels_to_census_block_groups as (
  select
    parcel_id
    , census_block_group_id
  from {{ref('parcels_to_census_block_groups')}}
)
select
  {{ dbt_utils.star(ref('parcels_base')) }}
  , zip_code_id
  , census_block_group_id
from
  {{ ref('parcels_base') }}
  left join parcels_to_zip_codes using (parcel_id)
  left join parcels_to_census_block_groups using (parcel_id)
