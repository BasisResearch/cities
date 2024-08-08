drop table if exists usps_migration_raw cascade;

create table usps_migration_raw (
    yyyymm text
    , zip_code text
    , city text
    , state text
    , total_from_zip numeric
    , total_from_zip_business numeric
    , total_from_zip_family numeric
    , total_from_zip_individual numeric
    , total_from_zip_perm numeric
    , total_from_zip_temp numeric
    , total_to_zip numeric
    , total_to_zip_business numeric
    , total_to_zip_family numeric
    , total_to_zip_individual numeric
    , total_to_zip_perm numeric
    , total_to_zip_temp numeric
    , year_ int
);

