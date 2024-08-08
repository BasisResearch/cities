drop type if exists parcel_census_bg_type cascade;

create type parcel_census_bg_type as enum (
    'within'
    , 'most_overlap'
    , 'closest'
);

drop table if exists parcel_census_bg;

create table parcel_census_bg (
    parcel_id int references parcel (id)
    , census_bg_id int references census_bg (id)
    , valid daterange not null
    , type parcel_census_bg_type not null
);

with parcel_with_geom as (
    select
        parcel.id
        , geom_id
        , valid
        , ST_Transform (geom
            , 4269) as geom
    from
        parcel
        join parcel_geom on geom_id = parcel_geom.id
)
, parcel_within as (
    -- easy case: one parcel in one bg
    select
        parcel.id as parcel_id
        , census_bg.id as census_bg_id
        , parcel.valid * census_bg.valid as valid
    from
        parcel_with_geom as parcel
        join census_bg on ST_Within (parcel.geom
            , census_bg.geom)
            and parcel.valid && census_bg.valid
)
, parcel_not_within as (
    -- parcels that are not fully within any bg
    select
        *
    from
        parcel_with_geom
    where
        not exists (
            select
                parcel_id
            from
                parcel_within
            where
                parcel_id = id)
)
, parcel_largest_overlap as (
    -- parcels that overlap multiple bgs map to the one with the largest overlap
    select distinct on (parcel.id)
        parcel.id as parcel_id
        , census_bg.id as census_bg_id
        , parcel.valid * census_bg.valid as valid
    from
        parcel_not_within as parcel
        join census_bg on ST_Intersects (parcel.geom
            , census_bg.geom)
            and parcel.valid && census_bg.valid
        order by
            parcel_id
            , ST_Area (ST_Intersection (parcel.geom
                    , census_bg.geom)) desc
)
, parcel_no_overlap as (
    -- parcels that do not overlap any bg
    select
        *
    from
        parcel_not_within
    where
        not exists (
            select
                parcel_id
            from
                parcel_largest_overlap
            where
                parcel_id = id)
)
, parcel_closest as (
    -- parcels that overlap no bgs map to the closest one
    select distinct on (parcel.id)
        parcel.id as parcel_id
        , census_bg.id as census_bg_id
        , parcel.valid * census_bg.valid as valid
    from
        parcel_no_overlap as parcel
        join census_bg on parcel.valid && census_bg.valid
    order by
        parcel_id
        , ST_Distance (parcel.geom
            , census_bg.geom))
    insert into parcel_census_bg
    select
        *
        , 'within'::parcel_census_bg_type
    from
        parcel_within
    union all
    select
        *
        , 'most_overlap'::parcel_census_bg_type
    from
        parcel_largest_overlap
    union all
    select
        *
        , 'closest'::parcel_census_bg_type
    from
        parcel_closest;

