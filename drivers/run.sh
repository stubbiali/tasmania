#!/bin/bash

NAMELISTS_FROM=(
	namelists/namelist_smolarkiewicz_cc_1.py
	namelists/namelist_smolarkiewicz_sus_1.py
	namelists/namelist_smolarkiewicz_ssus_1.py
	namelists/namelist_smolarkiewicz_cc_2.py
	namelists/namelist_smolarkiewicz_sus_2.py
	namelists/namelist_smolarkiewicz_ssus_2.py
	namelists/namelist_smolarkiewicz_cc_3.py
	namelists/namelist_smolarkiewicz_sus_3.py
	namelists/namelist_smolarkiewicz_ssus_3.py
)

NAMELISTS_TO=(
	namelist_smolarkiewicz_cc.py
	namelist_smolarkiewicz_sus.py
	namelist_smolarkiewicz_ssus.py
	namelist_smolarkiewicz_cc.py
	namelist_smolarkiewicz_sus.py
	namelist_smolarkiewicz_ssus.py
	namelist_smolarkiewicz_cc.py
	namelist_smolarkiewicz_sus.py
	namelist_smolarkiewicz_ssus.py
)

DRIVERS=(
	driver_namelist_smolarkiewicz_cc.py
	driver_namelist_smolarkiewicz_sus.py
	driver_namelist_smolarkiewicz_ssus.py
	driver_namelist_smolarkiewicz_cc.py
	driver_namelist_smolarkiewicz_sus.py
	driver_namelist_smolarkiewicz_ssus.py
	driver_namelist_smolarkiewicz_cc.py
	driver_namelist_smolarkiewicz_sus.py
	driver_namelist_smolarkiewicz_ssus.py
)

for i in $(seq 0 $((${#NAMELISTS_FROM[@]} - 1))); do
	printf "Copy %s into %s.\n" ${NAMELISTS_FROM[i]} ${NAMELISTS_TO[i]}
	cp ${NAMELISTS_FROM[i]} ${NAMELISTS_TO[i]}
	printf "Run %s.\n\n" ${DRIVERS[i]}
	python ${DRIVERS[i]}
	printf "\n%s completed.\n\n" ${DRIVERS[i]}
done
