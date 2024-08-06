drop table if exists residential_permit cascade;

create table residential_permit (
    id serial primary key,
    ctu_id text,
    coctu_id text,
    year int,
    tenure text,
    housing_ty text,
    res_permit text,
    address text,
    zip_code text,
    name text,
    buildings int,
    units int,
    age_restri int,
    memory_car int,
    assisted int,
    com_off_re boolean,
    sqf numeric,
    public_fun boolean,
    permit_val numeric,
    community_ text,
    notes text,
    pin text,
    geom geometry (multipoint, 26915)
);

create index residential_permit_geom_idx on residential_permit using gist (
    geom
);

insert into residential_permit (
    ctu_id,
    coctu_id,
    year,
    tenure,
    housing_ty,
    res_permit,
    address,
    zip_code,
    name,
    buildings,
    units,
    age_restri,
    memory_car,
    assisted,
    com_off_re,
    sqf,
    public_fun,
    permit_val,
    community_,
    notes,
    pin,
    geom
)
select
    ctu_id,
    coctu_id,
    year::int,
    tenure,
    housing_ty,
    res_permit,
    address,
    zip_code,
    name,
    buildings,
    units,
    age_restri,
    memory_car,
    assisted,
    com_off_re = 'Y',
    sqf,
    public_fun = 'Y',
    permit_val,
    community_,
    notes,
    pin,
    geom
from
    residential_permits_raw
where
    co_code = '053'
    and lower(ctu_name) = 'minneapolis';

drop table if exists residential_permit_parcel;

create table residential_permit_parcel (
    permit_id int references residential_permit (id),
    parcel_id int references parcel (id),
    type_ region_tag_type
);

with within as (
    select
        residential_permit.id as permit_id,
        parcel.id as parcel_id
    from
        parcel_with_geom as parcel
    join residential_permit on st_within(
        residential_permit.geom,
        parcel.geom
    )
    and to_date(
        year::text,
        'YYYY'
    ) <@ parcel.valid
),
not_within as (
    select
        id,
        year,
        geom
    from
        residential_permit
    where
        not exists (
            select permit_id
            from
                within
            where
                permit_id = id
        )
),
closest as (
    select distinct on (permit.id)
        permit.id as permit_id,
        parcel.id as parcel_id
    from
        not_within as permit
    join parcel_with_geom as parcel
        on st_dwithin(permit.geom, parcel.geom, 100.0) and to_date(
            year::text,
            'YYYY'
        ) <@ parcel.valid
    order by
        permit_id,
        st_distance(
            permit.geom,
            parcel.geom
        )
)
insert into residential_permit_parcel select
    permit_id,
    parcel_id,
    'within'::region_tag_type
from
    within
union all
select
    permit_id,
    parcel_id,
    'closest'::region_tag_type
from
    closest;

drop table if exists commercial_permit cascade;

create table commercial_permit (
    id serial primary key,
    ctu_id text,
    coctu_id text,
    year int,
    nonres_gro text,
    nonres_sub text,
    nonres_typ text,
    bldg_name text,
    bldg_desc text,
    permit_typ text,
    permit_val numeric,
    sqf int,
    address text,
    zip_code text,
    pin text,
    geom geometry (multipoint, 26915)
);

create index commercial_permit_geom_idx on commercial_permit using gist (
    geom
);

insert into commercial_permit (
    ctu_id,
    coctu_id,
    year,
    nonres_gro,
    nonres_sub,
    nonres_typ,
    bldg_name,
    bldg_desc,
    permit_typ,
    permit_val,
    sqf,
    address,
    zip_code,
    pin,
    geom
)
select
    ctu_id,
    coctu_id,
    year::int,
    nonres_gro,
    nonres_sub,
    nonres_typ,
    bldg_name,
    bldg_desc,
    permit_typ,
    permit_val,
    sqf,
    address,
    zip_code,
    pin,
    geom
from
    commercial_permits_raw
where
    co_code = '053'
    and lower(ctu_name) = 'minneapolis';

drop table if exists commercial_permit_parcel;

create table commercial_permit_parcel (
    permit_id int references commercial_permit (id),
    parcel_id int references parcel (id),
    type_ region_tag_type
);

with within as (
    select
        commercial_permit.id as permit_id,
        parcel.id as parcel_id
    from
        parcel_with_geom as parcel
    join commercial_permit on st_within(
        commercial_permit.geom,
        parcel.geom
    )
    and to_date(
        year::text,
        'YYYY'
    ) <@ parcel.valid
),
not_within as (
    select
        id,
        year,
        geom
    from
        commercial_permit
    where
        not exists (
            select permit_id
            from
                within
            where
                permit_id = id
        )
),
closest as (
    select distinct on (permit.id)
        permit.id as permit_id,
        parcel.id as parcel_id
    from
        not_within as permit
    join parcel_with_geom as parcel
        on st_dwithin(permit.geom, parcel.geom, 100.0) and to_date(
            year::text,
            'YYYY'
        ) <@ parcel.valid
    order by
        permit_id,
        st_distance(
            permit.geom,
            parcel.geom
        )
)
insert into commercial_permit_parcel select
    permit_id,
    parcel_id,
    'within'::region_tag_type
from
    within
union all
select
    permit_id,
    parcel_id,
    'closest'::region_tag_type
from
    closest;
