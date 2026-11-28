# Database Setup and Data Model

The Doctor Portal uses PostgreSQL for structured records and MongoDB Atlas GridFS for binary files. The prediction paths load local model artifacts and do not require a database connection.

## 1. Storage boundaries

| Data | Storage | Reason |
|---|---|---|
| Doctor profile | PostgreSQL | Structured fields and relational constraints |
| Password hash | PostgreSQL | Argon2id hash; plaintext password is not stored |
| Symptom contribution | PostgreSQL | Parent record and normalized symptom-item rows |
| Image contribution metadata | PostgreSQL | Disease label, submitter, file metadata, GridFS reference |
| Degree certificate PDF | MongoDB Atlas GridFS | Binary object in `doctor_degrees` bucket |
| Contribution image | MongoDB Atlas GridFS | Binary object in `contribution_images` bucket |

The databases are external managed services. They are not hosted on the application EC2 instance.

## 2. Runtime configuration

The database modules first read environment variables and fall back to Streamlit secrets.

| Environment variable | Streamlit secret | Required |
|---|---|---|
| `DISEASE_AKINATOR_POSTGRES_URL` | `[postgres].url` | Yes for Doctor Portal |
| `DISEASE_AKINATOR_MONGODB_URI` | `[mongodb].uri` | Yes for Doctor Portal |
| `DISEASE_AKINATOR_MONGODB_DATABASE` | `[mongodb].database` | Optional; defaults to `disease_akinator` |

A local `.env.example` is provided. Copy it to `.env` and replace the placeholders for a local environment. The populated `.env` and `.streamlit/secrets.toml` files must not be committed.

Production values are stored in AWS Systems Manager Parameter Store as a SecureString at `/disease-akinator/production/container-env`. Jenkins retrieves the parameter with decryption and passes it to the container at runtime. Never copy the actual values into documentation or logs.

## 3. PostgreSQL schema

Canonical schema file: `database/schema.sql`.

The schema creates the following application tables.

### `doctors`

Stores the submitted doctor profile and the reference/metadata for the uploaded degree PDF.

Important fields include:
- `id` — identity primary key.
- `full_name`, `age`, `degree`, `custom_degree`, `hospital`, `city`.
- `email` — normalized to lowercase/trimmed and unique.
- `password_hash` — Argon2id hash, not plaintext.
- `degree_file_id`, `degree_filename`, `degree_size_bytes`.
- `created_at`.

Constraints enforce required text, age range 25–80, supported degree values, normalized unique email, and the custom degree when `degree = 'Other'`.

### `symptom_contributions`

Stores one parent record for a doctor's disease/symptom submission:
- `id`
- `doctor_id` referencing `doctors`
- `disease_name`
- optional `notes`
- `submitted_at`

### `symptom_contribution_items`

Stores one row per symptom associated with a symptom contribution:
- `contribution_id`
- `position`
- `symptom_text`

The primary key preserves the symptom order within a contribution. A unique index prevents duplicate symptoms within one submission after case/whitespace normalization.

### `image_contributions`

Stores image contribution metadata:
- `id`
- `doctor_id`
- `disease_name`
- `mongo_file_id`
- `image_filename`
- `content_type`
- `file_size_bytes`
- optional `notes`
- `submitted_at`

The image bytes are not stored in this table. The `mongo_file_id` refers to the GridFS object.

### Indexes and permissions

The schema creates indexes for doctor/date, disease, and symptom lookups. It grants the `disease_akinator_app` role DML access to application tables and sequence usage/update access; it does not make the application role the schema owner.

The SQL file's default-privilege section references `neondb_owner`. Review that role name if using a different PostgreSQL owner. Apply schema changes with an administrative/owner role, not with the restricted application role.

## 4. Applying the schema

1. Create the `disease_akinator` database.
2. Create the schema owner role expected by the SQL script (or adapt the default-privilege role to your intended owner).
3. Create the restricted application role `disease_akinator_app`, or adapt the grant targets to the role you created.
4. Review the SQL file before running it.
5. Apply it with the database owner/admin connection:

       psql "<owner-connection-string>" -f database/schema.sql

6. Configure the application connection string using the application role, not the schema owner.
7. Verify the required tables and grants in the database console.

Do not put a real connection string in shell history, GitHub Actions logs, source code, or screenshots.

## 5. MongoDB Atlas and GridFS

Database name: `disease_akinator` by default.

GridFS buckets:

| Bucket | Content | Limit enforced by the app |
|---|---|---:|
| `doctor_degrees` | Doctor degree certificate PDFs | 10 MiB |
| `contribution_images` | Doctor-submitted labelled skin images | 5 MiB |

The application validates the file extension, file signature, and size before upload:
- Degree file: PDF, filename suffix `.pdf`, PDF signature `%PDF-`, maximum 10 MiB.
- Contribution image: PNG or JPEG, matching filename suffix/signature, maximum 5 MiB.

GridFS stores binary content and metadata. PostgreSQL stores the doctor/contribution records and the GridFS file identifier.

The MongoDB application user should have only the database permissions required for the application's GridFS operations. Restrict Atlas Network Access to the deployment's current egress address where feasible.

## 6. Cross-database upload lifecycle

The app uses two separate database services, so PostgreSQL and MongoDB do not participate in one shared ACID transaction.

### Doctor registration

1. Validate profile data and the PDF signature/size.
2. Hash the password using Argon2id.
3. Upload the PDF into `doctor_degrees`.
4. Insert the doctor profile in PostgreSQL with the file ID and metadata.
5. If the PostgreSQL insert fails, attempt to delete the uploaded GridFS PDF.
6. Return a safe profile without exposing the password hash.

### Image contribution

1. Validate the doctor ID, disease name, image type/signature, and size.
2. Upload the image into `contribution_images`.
3. Insert the image metadata and GridFS ID into PostgreSQL.
4. If the metadata insert fails, attempt a compensating delete of the GridFS object.
5. If cleanup cannot be confirmed, an operator may need to inspect the logged file reference and recover manually.

### Symptom contribution

1. Validate the doctor ID, disease name, and symptom list.
2. Insert a parent row in `symptom_contributions`.
3. Insert ordered symptom entries in `symptom_contribution_items`.

## 7. Authentication and limitations

- Password hashing uses Argon2id.
- Registration requires a password of at least 12 characters.
- The database stores the normalized email under a unique constraint.
- No email verification or password reset flow is implemented.
- The Doctor Portal does not independently verify professional credentials.
- Uploaded certificates are stored; they are not validated by a medical regulator.
- Contributions are not automatically used to retrain the symptom or image models.

## 8. Test and operational safety

- Use separate development/staging/test databases.
- Never run integration tests against production.
- Use synthetic records and non-identifying images for tests and demonstrations.
- Do not export production doctor profiles or degree PDFs into GitHub.
- Use database backups and retention controls appropriate to the provider and data owner; this repository does not define an automated backup/restore system.
- A healthy Streamlit container does not prove that the external databases are reachable. Test the Doctor Portal with a non-sensitive test account when validating a deployment.
