with
parcels as (select * from {{ ref('tracts_model_int__parcels_filtered') }}),
transit as (select * from {{ ref('high_frequency_transit_lines') }}),
downtown as (select * from {{ ref('downtown') }} limit 1),
university as (select * from {{ ref('university') }} limit 1),
with_is_downtown as (
  select
    parcels.parcel_id,
    parcels.census_tract_id,
    parcels.valid,
    parcels.geom,
    st_intersects(parcels.geom, downtown.geom) as is_downtown,
    st_intersects(parcels.geom, university.geom) as is_university
  from downtown, university, parcels
),
with_limit as (
  select
    parcels.parcel_id,
    parcels.census_tract_id,
    parcels.is_downtown,
    parcels.is_university,
    case
      when parcels.is_downtown then 'eliminated'
      when parcels.valid << '[2015-01-01,)'::daterange then 'full'
      else
        case
          when st_intersects(parcels.geom, transit.blue_zone_geom) then 'eliminated'
          when st_intersects(parcels.geom, transit.yellow_zone_geom) then 'reduced'
          else 'full'
        end
    end as limit_
  from
    with_is_downtown as parcels
    join transit on parcels.valid && transit.valid
),
with_limit_numeric as (
  select
    parcels.parcel_id,
    parcels.census_tract_id,
    parcels.is_downtown,
    parcels.is_university,
    parcels.limit_,
    case limit_
      when 'full' then 1
      when 'reduced' then 0.5
      when 'eliminated' then 0
    end as limit_numeric
  from with_limit as parcels
)
select * from with_limit_numeric
