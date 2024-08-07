drop table if exists acs_variable cascade;

create table acs_variable (
    name_ text primary key
    , description text not null
);

drop table if exists acs_tract_raw cascade;

create table acs_tract_raw (
    statefp text
    , countyfp text
    , tractce text
    , year_ int
    , name_ text
    , value_ numeric
);

drop table if exists acs_bg_raw cascade;

create table acs_bg_raw (
    statefp text
    , countyfp text
    , tractce text
    , blkgrpce text
    , year_ int
    , name_ text
    , value_ numeric
);

drop table if exists acs_tract cascade;

create table acs_tract (
    id int references census_tract (id)
    , year_ int not null
    , name_ text references acs_variable (name_)
    , value_ numeric
    , primary key (id , year_ , name_)
);

drop table if exists acs_bg cascade;

create table acs_bg (
    id int references census_bg (id)
    , year_ int not null
    , name_ text references acs_variable (name_)
    , value_ numeric
    , primary key (id , year_ , name_)
);

