db = db.getSiblingDB('exoplanet_ai');

db.createUser({
  user: 'exoplanet_user',
  pwd: 'exoplanet_pass',
  roles: [
    {
      role: 'readWrite',
      db: 'exoplanet_ai'
    }
  ]
});

db.createCollection('users');
db.createCollection('searches');
db.createCollection('feedback');

print('MongoDB initialization completed');