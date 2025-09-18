"""Create user authentication tables

Revision ID: 001_user_auth
Revises:
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001_user_auth'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Create user status enum
    user_status_enum = postgresql.ENUM(
        'ACTIVE', 'INACTIVE', 'SUSPENDED', 'PENDING_VERIFICATION', 'BANNED',
        name='userstatus',
        create_type=False
    )
    user_status_enum.create(op.get_bind())

    # Create user role enum
    user_role_enum = postgresql.ENUM(
        'USER', 'PREMIUM_USER', 'MODERATOR', 'ADMIN', 'SUPER_ADMIN',
        name='userrole',
        create_type=False
    )
    user_role_enum.create(op.get_bind())

    # Create auth provider enum
    auth_provider_enum = postgresql.ENUM(
        'LOCAL', 'GOOGLE', 'FACEBOOK', 'GITHUB', 'MICROSOFT',
        name='authprovider',
        create_type=False
    )
    auth_provider_enum.create(op.get_bind())

    # Create users table
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=True),
        sa.Column('full_name', sa.String(length=100), nullable=True),
        sa.Column('hashed_password', sa.String(length=255), nullable=True),
        sa.Column('is_password_set', sa.Boolean(), nullable=True),
        sa.Column('status', user_status_enum, nullable=True),
        sa.Column('role', user_role_enum, nullable=True),
        sa.Column('profile_image_url', sa.String(length=500), nullable=True),
        sa.Column('bio', sa.Text(), nullable=True),
        sa.Column('location', sa.String(length=100), nullable=True),
        sa.Column('website', sa.String(length=200), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('last_login_at', sa.DateTime(), nullable=True),
        sa.Column('last_activity_at', sa.DateTime(), nullable=True),
        sa.Column('is_email_verified', sa.Boolean(), nullable=True),
        sa.Column('email_verified_at', sa.DateTime(), nullable=True),
        sa.Column('preferences', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('settings', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_user_created_at', 'users', ['created_at'], unique=False)
    op.create_index('idx_user_email_status', 'users', ['email', 'status'], unique=False)
    op.create_index('idx_user_role_status', 'users', ['role', 'status'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    # Create user_sessions table
    op.create_table('user_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_token', sa.String(length=255), nullable=False),
        sa.Column('refresh_token', sa.String(length=255), nullable=True),
        sa.Column('device_id', sa.String(length=100), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('location', sa.String(length=100), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_session_expires_at', 'user_sessions', ['expires_at'], unique=False)
    op.create_index('idx_session_token', 'user_sessions', ['session_token'], unique=False)
    op.create_index('idx_session_user_active', 'user_sessions', ['user_id', 'is_active'], unique=False)
    op.create_index(op.f('ix_user_sessions_refresh_token'), 'user_sessions', ['refresh_token'], unique=True)
    op.create_index(op.f('ix_user_sessions_session_token'), 'user_sessions', ['session_token'], unique=True)

    # Create api_keys table
    op.create_table('api_keys',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('key_id', sa.String(length=50), nullable=False),
        sa.Column('key_hash', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('scopes', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('rate_limit_rpm', sa.Integer(), nullable=True),
        sa.Column('rate_limit_rph', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('usage_count', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_api_keys_key_id'), 'api_keys', ['key_id'], unique=True)

    # Create oauth_accounts table
    op.create_table('oauth_accounts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('provider', auth_provider_enum, nullable=False),
        sa.Column('provider_user_id', sa.String(length=255), nullable=False),
        sa.Column('provider_username', sa.String(length=100), nullable=True),
        sa.Column('provider_email', sa.String(length=255), nullable=True),
        sa.Column('access_token', sa.Text(), nullable=True),
        sa.Column('refresh_token', sa.Text(), nullable=True),
        sa.Column('token_expires_at', sa.DateTime(), nullable=True),
        sa.Column('profile_data', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_oauth_provider_user_id', 'oauth_accounts', ['provider', 'provider_user_id'], unique=True)
    op.create_index('idx_oauth_user_provider', 'oauth_accounts', ['user_id', 'provider'], unique=False)

    # Create login_attempts table
    op.create_table('login_attempts',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('ip_address', sa.String(length=45), nullable=False),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('failure_reason', sa.String(length=100), nullable=True),
        sa.Column('attempted_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_login_attempts_email_time', 'login_attempts', ['email', 'attempted_at'], unique=False)
    op.create_index('idx_login_attempts_ip_time', 'login_attempts', ['ip_address', 'attempted_at'], unique=False)
    op.create_index('idx_login_attempts_user_time', 'login_attempts', ['user_id', 'attempted_at'], unique=False)

    # Create permissions table
    op.create_table('permissions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('code', sa.String(length=100), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_permissions_code'), 'permissions', ['code'], unique=True)

    # Create roles table
    op.create_table('roles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('code', sa.String(length=50), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('level', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_roles_code'), 'roles', ['code'], unique=True)

    # Create user_permissions table
    op.create_table('user_permissions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('permission_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('granted', sa.Boolean(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('granted_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('granted_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['granted_by'], ['users.id'], ),
        sa.ForeignKeyConstraint(['permission_id'], ['permissions.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_user_permission_granted', 'user_permissions', ['user_id', 'granted'], unique=False)
    op.create_index('idx_user_permission_unique', 'user_permissions', ['user_id', 'permission_id'], unique=True)

    # Create role_permissions table
    op.create_table('role_permissions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('permission_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['permission_id'], ['permissions.id'], ),
        sa.ForeignKeyConstraint(['role_id'], ['roles.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_role_permission_unique', 'role_permissions', ['role_id', 'permission_id'], unique=True)

    # Create user_roles table
    op.create_table('user_roles',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('role_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('assigned_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('assigned_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(['assigned_by'], ['users.id'], ),
        sa.ForeignKeyConstraint(['role_id'], ['roles.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_user_role_active', 'user_roles', ['user_id', 'is_active'], unique=False)
    op.create_index('idx_user_role_unique', 'user_roles', ['user_id', 'role_id'], unique=True)

    # Create email verification tokens table
    op.create_table('email_verification_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('token', sa.String(length=255), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('is_used', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('used_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_email_verification_tokens_token'), 'email_verification_tokens', ['token'], unique=True)

    # Create password reset tokens table
    op.create_table('password_reset_tokens',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('token', sa.String(length=255), nullable=False),
        sa.Column('is_used', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('used_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_password_reset_tokens_token'), 'password_reset_tokens', ['token'], unique=True)

    # Insert default permissions
    op.execute("""
        INSERT INTO permissions (id, code, name, description, category, is_active, created_at) VALUES
        (gen_random_uuid(), 'fragrance.search', 'Search Fragrances', 'Search for fragrances', 'fragrance', true, now()),
        (gen_random_uuid(), 'fragrance.generate', 'Generate Fragrances', 'Generate new fragrances', 'fragrance', true, now()),
        (gen_random_uuid(), 'fragrance.save', 'Save Fragrances', 'Save fragrances to collection', 'fragrance', true, now()),
        (gen_random_uuid(), 'fragrance.share', 'Share Fragrances', 'Share fragrances with others', 'fragrance', true, now()),
        (gen_random_uuid(), 'user.profile.read', 'Read User Profile', 'View user profile information', 'user', true, now()),
        (gen_random_uuid(), 'user.profile.update', 'Update User Profile', 'Modify user profile information', 'user', true, now()),
        (gen_random_uuid(), 'admin.users.read', 'View Users', 'View user accounts and information', 'admin', true, now()),
        (gen_random_uuid(), 'admin.users.manage', 'Manage Users', 'Create, update, and delete user accounts', 'admin', true, now()),
        (gen_random_uuid(), 'admin.system.monitor', 'System Monitoring', 'View system metrics and health', 'admin', true, now()),
        (gen_random_uuid(), 'admin.system.configure', 'System Configuration', 'Configure system settings', 'admin', true, now()),
        (gen_random_uuid(), 'api.key.create', 'Create API Keys', 'Generate new API keys', 'api', true, now()),
        (gen_random_uuid(), 'api.key.manage', 'Manage API Keys', 'View and revoke API keys', 'api', true, now());
    """)

    # Insert default roles
    op.execute("""
        INSERT INTO roles (id, code, name, description, level, is_active, created_at) VALUES
        (gen_random_uuid(), 'user', 'User', 'Standard user with basic permissions', 0, true, now()),
        (gen_random_uuid(), 'premium_user', 'Premium User', 'Premium user with enhanced features', 1, true, now()),
        (gen_random_uuid(), 'moderator', 'Moderator', 'Content moderator with limited admin access', 2, true, now()),
        (gen_random_uuid(), 'admin', 'Administrator', 'Full administrative access', 3, true, now()),
        (gen_random_uuid(), 'super_admin', 'Super Administrator', 'Complete system access', 4, true, now());
    """)

    # Assign permissions to roles
    op.execute("""
        -- User role permissions
        INSERT INTO role_permissions (id, role_id, permission_id, created_at)
        SELECT gen_random_uuid(), r.id, p.id, now()
        FROM roles r, permissions p
        WHERE r.code = 'user' AND p.code IN (
            'fragrance.search', 'user.profile.read', 'user.profile.update', 'api.key.create'
        );

        -- Premium user role permissions (inherits user permissions)
        INSERT INTO role_permissions (id, role_id, permission_id, created_at)
        SELECT gen_random_uuid(), r.id, p.id, now()
        FROM roles r, permissions p
        WHERE r.code = 'premium_user' AND p.code IN (
            'fragrance.search', 'fragrance.generate', 'fragrance.save', 'fragrance.share',
            'user.profile.read', 'user.profile.update', 'api.key.create', 'api.key.manage'
        );

        -- Admin role permissions
        INSERT INTO role_permissions (id, role_id, permission_id, created_at)
        SELECT gen_random_uuid(), r.id, p.id, now()
        FROM roles r, permissions p
        WHERE r.code = 'admin' AND p.category IN ('fragrance', 'user', 'admin', 'api');

        -- Super admin gets all permissions
        INSERT INTO role_permissions (id, role_id, permission_id, created_at)
        SELECT gen_random_uuid(), r.id, p.id, now()
        FROM roles r, permissions p
        WHERE r.code = 'super_admin';
    """)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('password_reset_tokens')
    op.drop_table('email_verification_tokens')
    op.drop_table('user_roles')
    op.drop_table('role_permissions')
    op.drop_table('user_permissions')
    op.drop_table('roles')
    op.drop_table('permissions')
    op.drop_table('login_attempts')
    op.drop_table('oauth_accounts')
    op.drop_table('api_keys')
    op.drop_table('user_sessions')
    op.drop_table('users')

    # Drop enums
    op.execute('DROP TYPE IF EXISTS authprovider')
    op.execute('DROP TYPE IF EXISTS userrole')
    op.execute('DROP TYPE IF EXISTS userstatus')