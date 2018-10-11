#!/bin/bash

NAMELISTS=(
	namelists/namelist_rk3_51.py 
	namelists/namelist_rk3_101.py 
	namelists/namelist_rk3_201.py 
)

for namelist in ${NAMELISTS[*]}; do
	printf "Start %s.\n\n" $namelist
	cp $namelist namelist_isentropic_dry.py
	python driver_namelist_isentropic_dry_sus.py
	printf "\n%s done.\n\n" $namelist
done
