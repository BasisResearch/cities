with
commercial_permits_to_parcels as (
  select
    commercial_permit_id
    , parcel_id
  from {{ ref("commercial_permits_to_parcels") }}
)
select
  {{ dbt_utils.star(ref('commercial_permits_base')) }}
  , parcel_id
from
    {{ ref('commercial_permits_base') }}
    left join commercial_permits_to_parcels using (commercial_permit_id)
