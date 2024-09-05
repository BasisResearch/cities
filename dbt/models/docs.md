{% docs commercial_permits %}

Contains commercial building permit applications.

Notes:
 - Permits are filtered to only include those in Minneapolis.
 - `square_feet` is treated as missing if it is 0.
 - When mapping permits to parcels, if more than one parcel contains the permit
   location, a parcel will be chosen arbitrarily. This can happen because the
   same parcel spatial extent can appear multiple times with different PINs, to
   represent e.g. units in a condominium.

{% enddocs %}

{% docs residential_permits %}

Contains residential building permit applications.

Notes:
 - Permits are filtered to only include those in Minneapolis.
 - `square_feet` is treated as missing if it is 0.
 - `permit_value` is treated as missing if it is 0.
 - If more than one parcel contains the permit location, a parcel is selected
   arbitrarily. See `commercial_permits`.

{% enddocs %}

{% docs parking %}

Notes:
 - If more than one parcel contains the permit location, a parcel is selected
   arbitrarily. See `commercial_permits`.

{% enddocs %}

{% docs zctas %}

Contains the geometry and metadata for all zip code tabulation areas (ZCTAs) in
the United States.

These are not the same as zip codes. Zip codes are created by the postal service, and they change regularly. ZCTAs are created by the census bureau alongside the census. Not every zip code has a corresponding ZCTA (unpopulated zip codes are not represented, for example), and some ZCTAs cover multiple zip codes.

Use the mapping table `zip_codes_to_zctas` to translate from zip codes to ZCTAs.

{% enddocs %}

{% docs parcels %}

Contains the geometry and metadata for all parcels in the city of Minneapolis.

Notes:
- Parcels data is released yearly. Parcels are considered valid for the year they were released.
- Parcels are filtered to only include those in Minneapolis.
- `emv_total`, `emv_bldg`, `emv_land`, `year_built`, and `sale_value` are treated as missing if they are 0.
- `sale_date` is treated as missing if it is equal to `1899-12-30`.
- `pin` is the county-assigned parcel identification number. The county prefix '053-' is removed.
- Duplicate rows are removed. Note that this is based on the entire row, not just the `pin`. There may still be duplicate `pin, year_` pairs.

{% enddocs %}

{% docs census_tracts %}

Contains geometry and metadata for census tracts. Currently only includes census
tracts for Minnesota.

{% enddocs %}

{% docs census_block_groups %}

Contains geometry and metadata for census block groups. Currently only includes
census block groups for Minnesota.

{% enddocs %}

{% docs acs_block_group %}

Contains American Community Survey (ACS) demographic data at a census block
group granularity.

The `name_` column contains the name of the demographic variable (e.g.
`B03002_003E`). See `acs_variables` for a mapping of these codes to
human-readable names.

{% enddocs %}

{% docs acs_tract %}

Contains American Community Survey (ACS) demographic data at a census tract
granularity.

The `name_` column contains the name of the demographic variable (e.g.
`B03002_003E`). See `acs_variables` for a mapping of these codes to
human-readable names.

{% enddocs %}

{% docs fair_market_rents %}

Contains fair market rent data for different numbers of bedrooms by zip code.

{% enddocs %}

{% docs high_frequency_transit_lines %}

Contains the geometry and metadata for high frequency transit lines in the city of Minneapolis.

Notes:
- `blue_zone_geom` is a 350 foot buffer around both lines and stops.
- `yellow_zone_geom` is a quarter mile buffer around lines and a half mile buffer around stops.

{% enddocs %}

{% docs segregation_indexes %}

Segregation index for each tract for each year, computed for each reference
distribution.

The segregation index is the KL-divergence between the distribution of
population in a tract and a reference distribution. For example, a tract that
has many more white people than the average for the city will have a high
segregation index for the 'average_city' distribution.

Available distributions:
- `uniform`: Uniform distribution.
- `annual_city`: Citywide distribution for the current year.
- `average_city`: Citywide distribution averaged over all available years.

{% enddocs %}

{% docs usps_migration %}

Contains USPS migration data sourced from change of address forms. Migrations
are broken down by month and year, zip_code, flow direction, and flow type. Flow
directions are either `from` (out of) the zip code or `to` (in to) the zip code.

Flow types are one of `business`, `family`, `individual`, `perm` (permanent),
`temp` (temporary), or `total`.

We associate zip codes to ZCTAs and provide aggregate flows for ZCTAs. Note that
some zip codes do not find a match in our zip to ZCTA mapping table, so there is
some missingness in this data.

{% enddocs %}

{% docs demographics %}

Contains demographic data at census tract granularity.
Combines ACS data and segregation indexes in one table.

Notes:
- Fills in missing demographic data from 2011 and 2012 with data from 2013.
- Replaces pandemic-affected data from 2020 with data from 2019.

{% enddocs %}

{% docs neighborhoods %}

Neighborhood boundaries in the city of Minneapolis.

{% enddocs %}

{% docs wards %}

Ward boundaries in the city of Minneapolis.

{% enddocs %}

{% docs university %}

Boundary of the University of Minnesota.

{% enddocs %}

{% docs downtown %}

Boundary of the downtown of Minneapolis.

{% enddocs %}

{% docs city_boundary %}

Boundary of the city of Minneapolis.

{% enddocs %}