with
parcels as (
  select
    parcel_id as id
    , valid
    , geom
  from {{ ref("stg_parcels") }}
),
zctas as (
  select
    zcta_id as id
    , valid
    , geom
  from {{ ref("zctas") }}
)
select
  child_id as parcel_id
  , parent_id as zcta_id
  , valid
  , type_
from {{ tag_regions("parcels", "zctas") }}
