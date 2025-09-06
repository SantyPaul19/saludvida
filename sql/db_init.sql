-- Crea el esquema mínimo
create table if not exists patients (
    id bigserial primary key,
    created_at timestamptz not null default now(),
    ex_fum float8,
    consum_alcoh float8,
    consum_alcoh_30 float8,
    niveldeactividadesemanal float8,
    act_fis_frisk float8,
    diet_frisk float8,
    med_hta_fr float8,
    glu_alta float8,
    parien_dm float8,
    edad float8,
    mets float8,
    sedentarismo float8,
    talla float8,
    peso float8,
    imc float8,
    mme float8
);

create table if not exists predictions (
    id bigserial primary key,
    patient_id bigint references patients(id) on delete cascade,
    predicted_proba float8 not null,
    predicted_label int not null,
    threshold float8 not null,
    created_at timestamptz not null default now()
);

create or replace view latest_predictions as
select p.*, pr.predicted_proba, pr.predicted_label, pr.threshold, pr.created_at as predicted_at
from patients p
join lateral (
    select pr.* from predictions pr where pr.patient_id = p.id order by pr.created_at desc limit 1
) pr on true;
