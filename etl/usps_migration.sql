drop type if exists usps_migration_flow_direction cascade;

create type usps_migration_flow_direction as enum (
    'in'
    , 'out'
);

drop enum if exists usps_migration_flow_type cascade;

create type usps_migration_flow_type as enum (
    'total'
    , 'business'
    , 'family'
    , 'individual'
    , 'perm'
    , 'temp'
);

drop table if exists usps_migration cascade;

create table usps_migration (
    date_ date not null check (extract(day from date_) = 1) -- granularity is year-month
    , zip_id int references zip_code (id)
    , direction usps_migration_flow_direction not null
    , type_ usps_migration_flow_type not null
    , flow numeric
    , primary key (date_ , zip_id , direction , type_)
);

-- explain insert into usps_migration (date_, zip_id, direction, type_, flow)
insert into usps_migration with process_date as (
    select
        to_date(yyyymm
            , 'YYYYMM') as date_
        , *
    from
        usps_migration_raw
)
, add_zip_id as (
    select
        zip_code.id as zip_id
        , mr.*
    from
        process_date as mr
        join zip_code on zip_code.zip_code = replace(mr.zip_code
            , '='
            , '')
            and zip_code.valid @> to_date(year_::text
                , 'YYYY'))
    select
        date_
        , zip_id
        , 'in'::usps_migration_flow_direction
        , 'total'::usps_migration_flow_type
        , total_from_zip
    from
        add_zip_id
    union all
    select
        date_
        , zip_id
        , 'in'::usps_migration_flow_direction
        , 'business'::usps_migration_flow_type
        , total_from_zip_business
    from
        add_zip_id
    union all
    select
        date_
        , zip_id
        , 'in'::usps_migration_flow_direction
        , 'family'::usps_migration_flow_type
        , total_from_zip_family
    from
        add_zip_id
    union all
    select
        date_
        , zip_id
        , 'in'::usps_migration_flow_direction
        , 'individual'::usps_migration_flow_type
        , total_from_zip_individual
    from
        add_zip_id
    union all
    select
        date_
        , zip_id
        , 'in'::usps_migration_flow_direction
        , 'perm'::usps_migration_flow_type
        , total_from_zip_perm
    from
        add_zip_id
    union all
    select
        date_
        , zip_id
        , 'in'::usps_migration_flow_direction
        , 'temp'::usps_migration_flow_type
        , total_from_zip_temp
    from
        add_zip_id
    union all
    select
        date_
        , zip_id
        , 'out'::usps_migration_flow_direction
        , 'total'::usps_migration_flow_type
        , total_to_zip
    from
        add_zip_id
    union all
    select
        date_
        , zip_id
        , 'out'::usps_migration_flow_direction
        , 'business'::usps_migration_flow_type
        , total_to_zip_business
    from
        add_zip_id
    union all
    select
        date_
        , zip_id
        , 'out'::usps_migration_flow_direction
        , 'family'::usps_migration_flow_type
        , total_to_zip_family
    from
        add_zip_id
    union all
    select
        date_
        , zip_id
        , 'out'::usps_migration_flow_direction
        , 'individual'::usps_migration_flow_type
        , total_to_zip_individual
    from
        add_zip_id
    union all
    select
        date_
        , zip_id
        , 'out'::usps_migration_flow_direction
        , 'perm'::usps_migration_flow_type
        , total_to_zip_perm
    from
        add_zip_id
    union all
    select
        date_
        , zip_id
        , 'out'::usps_migration_flow_direction
        , 'temp'::usps_migration_flow_type
        , total_to_zip_temp
    from
        add_zip_id;

