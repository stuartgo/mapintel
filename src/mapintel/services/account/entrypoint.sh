#!/bin/sh

echo "Waiting for database PostgresSQL"

while ! nc -z db 5432; do
    sleep 0.1
done

echo "Database PostgresSQL started"

exec "$@"