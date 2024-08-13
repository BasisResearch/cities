with
residential_permits_to_parcels as (
  select
    residential_permit_id
    , parcel_id
  from {{ ref("residential_permits_to_parcels") }}
)
select
  {{ dbt_utils.star(ref('residential_permits_base')) }}
  , parcel_id
from
    {{ ref('residential_permits_base') }}
    left join residential_permits_to_parcels using (residential_permit_id)
