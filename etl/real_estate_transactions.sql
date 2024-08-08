drop table if exists real_estate_transactions_scraped;

create table real_estate_transactions_scraped (
    parcel_id text
    , address text
    , sale_date date
    , sale_price numeric
    , building_area numeric
    , beds numeric
    , baths numeric
    , stories numeric
    , year_built numeric
    , neighborhood text
    , property_type text
);

\copy real_estate_transactions_scraped from 'zoning/data/processed/real_estate_transactions/real_estate_transactions.csv' with csv header delimiter ',';
drop table if exists real_estate_transactions_raw;

create table real_estate_transactions_raw (
    sale_id int
    , ecrv text
    , sale_date date
    , excluded_from_ratio_study text
    , pin text
    , num_parcels_in_sale int
    , formatted_address text
    , land_sale text
    , community_cd int
    , community_desc text
    , nbhd_cd int
    , nbhd_desc text
    , ward int
    , proptype_cd text
    , proptype_desc text
    , grantee1 text
    , grantee2 text
    , grantor1 text
    , grantor2 text
    , adj_sale_price int
    , gross_sale_price int
    , downpayment int
    , x numeric
    , y numeric
    , fid int
);

\copy real_estate_transactions_raw from 'zoning/data/raw/real_estate_transactions/Property_Sales_2019_to_2023.csv' with csv header delimiter ',';
drop table if exists real_estate_transactions;

create table real_estate_transactions (
    id serial primary key
    , parcel_id int references parcel (id)
    , address text
    , sale_date date
    , sale_price numeric
    , neighborhood text
    , property_type text
);

insert into real_estate_transactions (parcel_id , address , sale_date , sale_price , neighborhood , property_type)
select
    parcel.id
    , address
    , scraped.sale_date
    , sale_price
    , neighborhood
    , property_type
from
    real_estate_transactions_scraped as scraped
    join parcel on pid = parcel_id
        and scraped.sale_date <@ valid;

