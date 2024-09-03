with
parcels as (
  select
    parcel_id as id
    , valid
    , geom
  from {{ ref("stg_parcels_base") }}
),
zip_codes as (
  select
    zip_code_id as id
    , valid
    , geom
  from {{ ref("zip_codes") }}
)
select
  child_id as parcel_id
  , parent_id as zip_code_id
  , valid
  , type_
from {{ tag_regions("parcels", "zip_codes") }}
