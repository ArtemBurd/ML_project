CREATE TABLE IF NOT EXISTS bakery_sales (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    income FLOAT,
    croissant INTEGER,
    tartelette INTEGER,
    boisson_33cl INTEGER
);