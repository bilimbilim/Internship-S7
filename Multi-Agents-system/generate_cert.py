#!/usr/bin/env python3
"""
Script pour générer des certificats SSL auto-signés pour HTTPS
"""

from OpenSSL import crypto
import os

def generate_self_signed_cert():
    """Génère un certificat SSL auto-signé"""
    
    # Créer une paire de clés
    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 2048)
    
    # Créer un certificat auto-signé
    cert = crypto.X509()
    cert.get_subject().C = "FR"
    cert.get_subject().ST = "France"
    cert.get_subject().L = "Paris"
    cert.get_subject().O = "Agent Email IA"
    cert.get_subject().OU = "Development"
    cert.get_subject().CN = "localhost"
    
    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365*24*60*60)  # Valide 1 an
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)
    cert.sign(key, 'sha256')
    
    # Sauvegarder le certificat et la clé
    with open("cert.pem", "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    
    with open("key.pem", "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
    
    print(" Certificats SSL générés :")
    print("   - cert.pem")
    print("   - key.pem")
    print("")
    print("  ATTENTION : Ce sont des certificats auto-signés.")
    print("   Votre navigateur affichera un avertissement de sécurité.")
    print("   Pour la production, utilisez Let's Encrypt ou un certificat valide.")

if __name__ == "__main__":
    if os.path.exists("cert.pem") or os.path.exists("key.pem"):
        response = input("Des certificats existent déjà. Écraser ? (o/n) : ")
        if response.lower() != 'o':
            print("Opération annulée.")
            exit(0)
    
    generate_self_signed_cert()
