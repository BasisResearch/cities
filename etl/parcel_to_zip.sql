drop type if exists parcel_zip_type;
create type parcel_zip_type as enum ('within', 'most_overlap', 'closest');

drop table if exists parcel_zip;
create table parcel_zip (
  parcel_id int references parcel(id)
  , zip_code_id int references zip_code(id)
  , valid daterange not null
  , type parcel_zip_type not null
);

with
parcel_with_geom as (
     select parcel.id, geom_id, valid, ST_Transform(geom, 4269) as geom
     from parcel
     join parcel_geom on geom_id = parcel_geom.id
),
parcel_in_zip as ( -- easy case: one parcel in one zip code
     select parcel.id as parcel_id,
            zip_code.id as zip_code_id,
            parcel.valid * zip_code.valid as valid
     from parcel_with_geom as parcel
     join zip_code on ST_Within(parcel.geom, zip_code.geom) and parcel.valid && zip_code.valid
),
parcel_not_within_zip as ( -- parcels that are not fully within any zip code
     select *
     from parcel_with_geom
     where not exists (select parcel_id from parcel_in_zip where parcel_id = id)
),
parcel_largest_overlap as ( -- parcels that overlap multiple zip codes map to the one with the largest overlap
     select distinct on (parcel.id)
            parcel.id as parcel_id,
            zip_code.id as zip_code_id,
            parcel.valid * zip_code.valid as valid
     from parcel_not_within_zip as parcel
     join zip_code on ST_Intersects(parcel.geom, zip_code.geom) and parcel.valid && zip_code.valid
     order by parcel_id, ST_Area(ST_Intersection(parcel.geom, zip_code.geom)) desc
),
parcel_no_overlap as ( -- parcels that do not overlap any zip code
     select *
     from parcel_not_within_zip
     where not exists (select parcel_id from parcel_largest_overlap where parcel_id = id)
),
parcel_closest as ( -- parcels that overlap no zip codes map to the closest one
     select distinct on (parcel.id)
            parcel.id as parcel_id,
            zip_code.id as zip_code_id,
            parcel.valid * zip_code.valid as valid
     from parcel_no_overlap as parcel
     join zip_code on parcel.valid && zip_code.valid
     order by parcel_id, ST_Distance(parcel.geom, zip_code.geom)
)
insert into parcel_zip
select *, 'within'::parcel_zip_type from parcel_in_zip
union all
select *, 'most_overlap'::parcel_zip_type from parcel_largest_overlap
union all
select *, 'closest'::parcel_zip_type from parcel_closest;
