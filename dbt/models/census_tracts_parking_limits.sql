{{
  config(
    materialized='table',
    indexes = [
      {'columns': ['census_tract_id'], 'unique': true},
    ]
  )
}}

with
parcels as (select * from {{ ref('parcels') }}),
transit as (select * from {{ ref('high_frequency_transit_lines') }}),
downtown as (select * from {{ ref('downtown') }}),
with_parking_limit as (
  select
    parcel_id,
    census_tract_id,
    case
      when st_intersects(parcels.geom, downtown.geom) then 'eliminated'
      when parcels.valid << '[2015-01-01,)'::daterange then 'full'
      else
        case
          when st_intersects(parcels.geom, transit.blue_zone_geom) then 'eliminated'
          when st_intersects(parcels.geom, transit.yellow_zone_geom) then 'reduced'
          else 'full'
        end
    end as limit_
  from
    downtown, parcels
    left join transit
      on parcels.valid && transit.valid
),
with_limit_numeric as (
  select
    parcel_id,
    census_tract_id,
    limit_,
    case limit_
      when 'full' then 1
      when 'reduced' then 0.5
      when 'eliminated' then 0
    end as limit_numeric
  from with_parking_limit
),
by_census_tract as (
  select
    census_tract_id,
    avg(limit_numeric) as mean_limit
  from with_limit_numeric
  group by census_tract_id
)
select * from by_census_tract
