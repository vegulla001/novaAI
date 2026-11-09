import hmac
from dataclasses import dataclass
import jwt
from fastapi import Header, HTTPException
from .config import settings

@dataclass(frozen=True)
class Principal:
    tenant: str
    user: str

_jwks_clients = {}

def principal(authorization: str | None = Header(default=None)) -> Principal:
    s = settings()
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Bearer token required")
    token = authorization[7:]
    if s.dev_auth_enabled and hmac.compare_digest(token, s.dev_api_token):
        return Principal(tenant="dev", user="developer")
    if not all([s.oidc_issuer, s.oidc_audience, s.oidc_jwks_url]):
        raise HTTPException(401, "OIDC authentication not configured")
    try:
        client = _jwks_clients.setdefault(s.oidc_jwks_url, jwt.PyJWKClient(s.oidc_jwks_url))
        key = client.get_signing_key_from_jwt(token).key
        claims = jwt.decode(token, key, algorithms=["RS256"], issuer=s.oidc_issuer,
                            audience=s.oidc_audience, options={"require": ["exp", "iat", "sub", "iss", "aud"]})
        user = claims["sub"]
        tenant = claims.get("tenant_id")
        if not isinstance(user, str) or not user or not isinstance(tenant, str) or not tenant:
            raise ValueError("missing tenant_id/sub")
        return Principal(tenant=tenant, user=user)
    except (jwt.PyJWTError, ValueError) as exc:
        raise HTTPException(401, "Invalid access token") from exc
