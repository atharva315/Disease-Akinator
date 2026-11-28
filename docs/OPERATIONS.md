# Production Operations Runbook

This runbook covers routine checks for the single EC2-hosted production deployment. It is intentionally limited to the Disease Akinator instance. Do not use these instructions on the separate ResumeClassification EC2 instance.

## 1. Production components

- EC2 instance: `disease-akinator-server`
- Region: `ap-southeast-2` (Sydney)
- Public hostname: `https://disease-akinator.duckdns.org`
- Application container: `disease-akinator`
- Container image repository: Amazon ECR `disease-akinator`
- Internal Streamlit port: `127.0.0.1:8501`
- Public reverse proxy: Nginx on ports 80/443
- Databases: external Neon PostgreSQL and MongoDB Atlas

## 2. Start the EC2 host

1. Open AWS Console → EC2 → Instances.
2. Select only `disease-akinator-server`.
3. Choose **Instance state → Start instance**.
4. Wait for the instance to become Running and for both EC2 status checks to pass.
5. Record the new Public IPv4 address.

The instance uses an automatically assigned public IPv4 address, not a reserved Elastic IP. Its address can change after a stop/start.

## 3. Refresh DNS and MongoDB Atlas access

After every start:

1. Open DuckDNS and update the `disease-akinator` record to the new EC2 public IPv4 address.
2. Open MongoDB Atlas → Security → Network Access.
3. Add the new EC2 address as a single-host `/32` entry.
4. Keep the previous entry until the new deployment has been verified; stale entries can be reviewed and removed later according to the operator's policy.
5. Allow DNS changes to resolve before testing the public site.

Do not use `0.0.0.0/0` as a shortcut for Atlas access.

## 4. Verify services after boot

Connect by SSH using the current EC2 public IP/DNS and the operator's private key. Then run:

    sudo systemctl is-active docker nginx jenkins
    docker ps

Expected:
- `docker`, `nginx`, and `jenkins` report `active`.
- The `disease-akinator` container is listed.
- The application container reports healthy after startup.

For a concise container inspection:

    docker inspect --format='Image={{.Config.Image}} | Status={{.State.Status}} | Health={{.State.Health.Status}}' disease-akinator

The container is configured with Docker restart policy `unless-stopped`; Docker, Nginx, and Jenkins are configured as services enabled at boot.

## 5. Verify local and public application health

From the EC2 shell:

    curl -I --max-time 10 -H "Host: disease-akinator.duckdns.org" http://127.0.0.1
    curl -kI --max-time 10 --resolve disease-akinator.duckdns.org:443:127.0.0.1 https://disease-akinator.duckdns.org/

Expected local HTTP behaviour is a redirect to HTTPS. Local HTTPS should return an HTTP success response from Nginx.

From a client computer, check DNS and port reachability:

    nslookup disease-akinator.duckdns.org
    Test-NetConnection <current-public-ip> -Port 443
    Test-NetConnection <current-public-ip> -Port 80

Then open `https://disease-akinator.duckdns.org` in the browser. A hard refresh or Incognito window can help rule out stale browser state.

## 6. Jenkins access

Jenkins is not exposed publicly; use an SSH local port forward from Windows PowerShell:

    ssh -i "<path-to-private-key.pem>" -N -L 8080:127.0.0.1:8080 ec2-user@<current-public-ip-or-dns>

Keep that terminal open and browse to:

    http://localhost:8080

Use the current EC2 address after a restart. Closing the SSH tunnel closes browser access to Jenkins but does not stop Jenkins itself.

## 7. Application and service logs

Application container logs:

    docker logs --tail 100 disease-akinator
    docker logs --tail 100 -f disease-akinator

Nginx service and recent logs:

    sudo systemctl status nginx --no-pager
    sudo journalctl -u nginx -n 100 --no-pager
    sudo nginx -t

Jenkins service and recent logs:

    sudo systemctl status jenkins --no-pager
    sudo journalctl -u jenkins -n 100 --no-pager

Do not paste logs publicly until checking that they contain no secret values, private records, or identifying data.

## 8. Common troubleshooting

### Browser reports a timeout

1. Check the current EC2 public IP.
2. Confirm DuckDNS resolves to that IP.
3. Confirm inbound Security Group rules allow HTTP 80 and HTTPS 443 from intended public clients.
4. Confirm Nginx is active and its configuration passes `sudo nginx -t`.
5. Test local HTTP/HTTPS from EC2.
6. Test TCP connectivity from the client.
7. If local Nginx returns success and external TCP works, retry with a hard refresh/Incognito and inspect client network/proxy behaviour.

### Domain resolves to an old IP

Update the DuckDNS record to the current EC2 public IPv4 address and allow DNS caches to expire. Do not hard-code an old IP into this documentation.

### Doctor Portal cannot reach MongoDB after a restart

Check that the new EC2 egress/public IP was added to MongoDB Atlas Network Access as a `/32` entry. Then test the database connection from the app environment without printing the URI.

### Docker container is not healthy

    docker ps -a
    docker inspect disease-akinator
    docker logs --tail 200 disease-akinator

Check Nginx and Docker service status as well. A healthy container verifies the Streamlit health endpoint; it does not prove external database connectivity or model prediction quality.

### Jenkins reports a GitHub commit-status 403 warning

A read-only GitHub token cannot write commit status. Check whether the actual stages succeeded and whether Jenkins ends with `Finished: SUCCESS`. Do not treat the status-write warning alone as a failed deployment.

### Jenkins did not notice a release

The Jenkinsfile polls every five minutes while Jenkins is running. Confirm EC2/Jenkins is up, open the multibranch pipeline, and inspect the main branch's polling/build history. A stopped host cannot poll while offline.

## 9. Stop EC2 when idle

Stopping the instance is supported when the application is not needed:

1. AWS Console → EC2 → Instances.
2. Select only `disease-akinator-server`.
3. Choose **Instance state → Stop instance** (never Terminate for routine shutdown).
4. Wait for Stopped.

While stopped, the application and Jenkins are offline; GitHub Actions remains available. EBS storage and retained AWS resources may continue to incur charges. The public IPv4 address is released and can change on the next start. Repeat the DNS and Atlas allowlist steps after restarting.

Neon PostgreSQL, MongoDB Atlas, ECR images, GitHub source, and SSM parameters are external to the stopped EC2 host and are not deleted by stopping it.
