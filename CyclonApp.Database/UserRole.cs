using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CyclonApp.Database
{
    public class UserRole
    {
        public int Id { get; set; }
        public string RoleName { get; set; } = string.Empty;
        public bool IsActive { get; set; }

        // Navigation
        public ICollection<AppUser> Users { get; set; } = new List<AppUser>();

    }
  
}
